// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file eliminate_simdvf_roundtrip.cc
 * \brief EliminateSimdVFRoundTrip: remove provably-safe UB round-trip buffers
 * inside PTO SimdVF regions before they are vectorized.
 *
 * A "round trip" is a region-local buffer that only shuttles a value through
 * UB inside one SIMD_VF region: unit A stores it, a later unit loads it back.
 * After VectorizeParallelToPTO such a buffer becomes a vstore followed by a
 * vload in the same vecscope, and the machine-level schedule does not honour
 * that UB RAW (the load may observe the previous UB contents). The value
 * never needed UB in the first place, so this pass removes the buffer while
 * the IR is still scalar parallel TIR, which keeps the vectorizer (and VMI)
 * unchanged.
 *
 * Pipeline position: `VerifyParallelToPTO -> LegalizeParallelToPTO ->
 * EliminateSimdVFRoundTrip -> VectorizeParallelToPTO` (PTO target only).
 *
 * Two conservative rules, applied to top-level sequential units only:
 *
 *  - Store forwarding: `T[i] = expr` followed by the immediately
 *    adjacent pure copy `OUT[i] = T[i]` becomes `OUT[i] = expr` and the copy
 *    unit is deleted. `T` must be region-local, written only by the producer
 *    and read only by that copy; `OUT` must have no other writer and no read
 *    inside the region.
 *  - Copy propagation: the pure copy producer
 *    `T[i] = S[i]` followed by the immediately adjacent consumer unit that
 *    loads `T[i]` becomes a direct read of `S[i]`, and the copy unit is
 *    deleted. `T` must be region-local, single-writer and single-reader;
 *    `S` must not be written anywhere in the region so the value observed by
 *    the consumer is the value the copy would have captured.
 *
 * Adjacency (no statement between the producer and the consumer) is imposed
 * on purpose. It makes the rewrite independent of any UB aliasing decision
 * taken earlier in the pipeline (MergeUBAllocations runs much later, but
 * `tl.buffer_alias_map` already relates buffers in autoscheduled input): with
 * nothing in between, moving the store one statement earlier and replacing a
 * snapshot load by a reread cannot cross another access to the same storage.
 *
 * Everything outside this whitelist is left untouched: buffers declared
 * outside the region, buffers read outside the region, extra writers or
 * readers, non-identity indices, masked accesses, 2D units, units under a
 * serial wrapper (loop-carried values must stay in UB) and units with a tail
 * (non-divisible extent) are all skipped. The pass is idempotent and its
 * output stays a legal input for VerifyParallelToPTO.
 *
 * Deleting a unit may leave its `alloc_buffers` entry without any remaining
 * use in the region; such entries are pruned so MergeUBAllocations (which
 * runs later) does not reserve UB for dead round-trip storage.
 */

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parallel_to_pto_utils.h"
#include "pto_slot_plan.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
namespace pto_analysis = ::tvm::tl::pto;

namespace {

Stmt NoOpStmt() { return Evaluate(make_zero(DataType::Int(32))); }

void FlattenSeq(const Stmt &stmt, std::vector<Stmt> *out) {
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      FlattenSeq(s, out);
    }
    return;
  }
  out->push_back(stmt);
}

/*! The SIMD_VF region body as an ordered list of top-level statements with
 * the `tl.simdvf_scope` attribute wrapper removed. */
std::vector<Stmt> RegionStatements(const Stmt &region) {
  const Stmt *inner = &region;
  if (const auto *attr = region.as<AttrStmtNode>()) {
    if (attr->attr_key == "tl.simdvf_scope") {
      inner = &attr->body;
    }
  }
  std::vector<Stmt> out;
  FlattenSeq(*inner, &out);
  return out;
}

bool IsIdentityIndex(const Array<PrimExpr> &indices, const VarNode *loop_var) {
  if (loop_var == nullptr || indices.size() != 1) {
    return false;
  }
  const auto *var = indices[0].as<VarNode>();
  return var != nullptr && var == loop_var;
}

/*! Access bookkeeping for one buffer data variable over a whole region. */
struct BufferAccessRecord {
  int stores = 0;
  int loads = 0;
  /*! Uses of the data handle that are neither a BufferLoad nor a
   * BufferStore (e.g. handed to an intrinsic). Any such use disqualifies the
   * buffer. */
  int other = 0;
  bool predicated = false;
  bool loads_identity = true;
  bool stores_identity = true;
  /*! Top-level unit ordinal owning each store/load; -1 means "not a
   * top-level parallel unit" (serial wrapper, plain statement, ...). */
  std::vector<int> store_units;
  std::vector<int> load_units;
};

/*! What one top-level parallel unit looks like to the two rules. */
struct UnitShape {
  bool exists = false;
  size_t stmt_pos = 0;
  int ordinal = -1;
  const VarNode *loop_var = nullptr;
  /*! True when the body is a sequence of Binds plus exactly one BufferStore
   * (no other side effect). */
  bool one_store = false;
  const BufferStoreNode *store = nullptr;
  bool store_identity = false;
  bool store_unpredicated = false;
  /*! True when that store is the pure copy `target[i] = source[i]`. */
  bool pure_copy = false;
  Buffer copy_source;
  bool one_d = false;
  bool divisible = false;
  int64_t extent = 0;
};

struct RegionModel {
  const AttrStmtNode *scope = nullptr;
  std::vector<Stmt> stmts;
  std::vector<UnitShape> units;
  std::unordered_map<const VarNode *, BufferAccessRecord> access;
};

/*! Collect the accesses of one top-level statement, tagging them with the
 * owning unit ordinal (or -1). */
class StmtAccessCollector final : public StmtExprVisitor {
 public:
  StmtAccessCollector(const VarNode *loop_var, int ordinal,
                      const std::unordered_set<const VarNode *> *data_vars,
                      std::unordered_map<const VarNode *, BufferAccessRecord> *out)
      : loop_var_(loop_var), ordinal_(ordinal), data_vars_(*data_vars), out_(out) {}

  void VisitStmt_(const BufferStoreNode *op) final {
    BufferAccessRecord &rec = (*out_)[op->buffer->data.get()];
    ++rec.stores;
    rec.store_units.push_back(ordinal_);
    if (op->predicate.defined()) {
      rec.predicated = true;
    }
    if (!IsIdentityIndex(op->indices, loop_var_)) {
      rec.stores_identity = false;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    BufferAccessRecord &rec = (*out_)[op->buffer->data.get()];
    ++rec.loads;
    rec.load_units.push_back(ordinal_);
    if (op->predicate.defined()) {
      rec.predicated = true;
    }
    if (!IsIdentityIndex(op->indices, loop_var_)) {
      rec.loads_identity = false;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    if (data_vars_.count(op) != 0) {
      ++(*out_)[op].other;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  /*! Public entry: VisitStmt is protected in StmtVisitor. */
  void Run(const Stmt &stmt) { VisitStmt(stmt); }

 private:
  const VarNode *loop_var_;
  int ordinal_;
  const std::unordered_set<const VarNode *> &data_vars_;
  std::unordered_map<const VarNode *, BufferAccessRecord> *out_;
};

UnitShape MakeUnitShape(const ForNode *loop, size_t stmt_pos, int ordinal) {
  UnitShape shape;
  shape.exists = true;
  shape.stmt_pos = stmt_pos;
  shape.ordinal = ordinal;
  shape.loop_var = loop->loop_var.get();
  std::vector<Stmt> body;
  FlattenSeq(loop->body, &body);
  for (const Stmt &s : body) {
    if (const auto *store = s.as<BufferStoreNode>()) {
      if (shape.store != nullptr) {
        // More than one store: neither rule may touch this unit.
        shape.store = nullptr;
        shape.one_store = false;
        return shape;
      }
      shape.store = store;
    } else if (s.as<BindNode>() == nullptr) {
      shape.store = nullptr;
      shape.one_store = false;
      return shape;
    }
  }
  if (shape.store == nullptr) {
    return shape;
  }
  shape.one_store = true;
  shape.store_unpredicated = !shape.store->predicate.defined();
  shape.store_identity = IsIdentityIndex(shape.store->indices, shape.loop_var);
  if (shape.store_unpredicated && shape.store_identity) {
    const auto *load = shape.store->value.as<BufferLoadNode>();
    if (load != nullptr && !load->predicate.defined() &&
        IsIdentityIndex(load->indices, shape.loop_var) &&
        load->buffer->data.get() != shape.store->buffer->data.get() &&
        load->buffer->dtype == shape.store->buffer->dtype) {
      shape.pure_copy = true;
      shape.copy_source = load->buffer;
    }
  }
  // 1D / divisibility facts come from the shared inverse mapping so this pass
  // classifies units exactly like Verify and Vectorize.
  try {
    pto_analysis::PtoInverseMapping mapping =
        pto_analysis::InvertPtoLaneLayout(GetRef<For>(loop));
    shape.one_d = !mapping.is_2d;
    shape.extent = mapping.logical_extent;
    shape.divisible = mapping.tail_lanes == 0;
  } catch (const std::exception &) {
    shape.one_d = false;
  }
  return shape;
}

RegionModel BuildRegionModel(const Stmt &region) {
  RegionModel model;
  model.stmts = RegionStatements(region);
  if (const auto *attr = region.as<AttrStmtNode>()) {
    if (attr->attr_key == "tl.simdvf_scope") {
      model.scope = attr;
    }
  }

  // Buffer data variables that take part in an access somewhere in the
  // region: a raw use of one of them is an escape the rules must not cross.
  std::unordered_set<const VarNode *> data_vars;
  PostOrderVisit(region, [&data_vars](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) {
      data_vars.insert(load->buffer->data.get());
    } else if (const auto *store = node.as<BufferStoreNode>()) {
      data_vars.insert(store->buffer->data.get());
    }
  });

  int ordinal = 0;
  for (size_t pos = 0; pos < model.stmts.size(); ++pos) {
    const Stmt &stmt = model.stmts[pos];
    const ForNode *parallel = nullptr;
    if (const auto *for_node = stmt.as<ForNode>()) {
      if (for_node->kind == ForKind::kParallel) {
        parallel = for_node;
      }
    }
    if (parallel != nullptr) {
      model.units.push_back(MakeUnitShape(parallel, pos, ordinal));
      StmtAccessCollector collector(parallel->loop_var.get(), ordinal, &data_vars,
                                    &model.access);
      collector.Run(stmt);
      ++ordinal;
    } else {
      StmtAccessCollector collector(nullptr, -1, &data_vars, &model.access);
      collector.Run(stmt);
    }
  }
  return model;
}

const UnitShape *UnitAt(const RegionModel &model, size_t stmt_pos) {
  for (const UnitShape &unit : model.units) {
    if (unit.exists && unit.stmt_pos == stmt_pos) {
      return &unit;
    }
  }
  return nullptr;
}

Stmt RebuildRegion(const AttrStmtNode *scope, const std::vector<Stmt> &stmts) {
  Stmt body;
  if (stmts.empty()) {
    body = NoOpStmt();
  } else if (stmts.size() == 1) {
    body = stmts[0];
  } else {
    Array<Stmt> seq;
    seq.reserve(static_cast<size_t>(stmts.size()));
    for (const Stmt &s : stmts) {
      seq.push_back(s);
    }
    body = SeqStmt(seq);
  }
  if (scope != nullptr) {
    body = AttrStmt(scope->node, scope->attr_key, scope->value, body);
  }
  return body;
}

/*! Replace the one matched store's destination buffer, keeping value,
 * indices and predicate. */
class StoreRetargeter final : public StmtExprMutator {
 public:
  StoreRetargeter(const BufferStoreNode *target, Buffer to)
      : target_(target), to_(std::move(to)) {}

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (op == target_) {
      return BufferStore(to_, op->value, op->indices, op->predicate, op->span);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  /*! Public entry: VisitStmt is protected in StmtMutator. */
  Stmt Apply(const Stmt &stmt) { return VisitStmt(stmt); }

 private:
  const BufferStoreNode *target_;
  Buffer to_;
};

/*! Rewrite every load of one buffer data variable into a load of another
 * buffer, keeping the index expression unchanged (identity indices only, so
 * the replacement is the same element of the same-shaped buffer). */
class LoadRetargeter final : public StmtExprMutator {
 public:
  LoadRetargeter(const VarNode *from, Buffer to) : from_(from), to_(std::move(to)) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer->data.get() == from_) {
      return BufferLoad(to_, op->indices, op->predicate, op->span);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  /*! Public entry: VisitStmt is protected in StmtMutator. */
  Stmt Apply(const Stmt &stmt) { return VisitStmt(stmt); }

 private:
  const VarNode *from_;
  Buffer to_;
};

/*! Whole-function scan: for every buffer data variable, which SIMD_VF blocks
 * touch it and whether anything outside a SIMD_VF block does. */
class RegionLocalityScan final : public StmtExprVisitor {
 public:
  void Scan(const Stmt &body) { VisitStmt(body); }

  bool LocalTo(const VarNode *var, const SBlockNode *block) const {
    if (outside_.count(var) != 0) {
      return false;
    }
    auto it = blocks_.find(var);
    if (it == blocks_.end()) {
      return true;  // never accessed
    }
    return it->second.size() == 1 && it->second.count(block) == 1;
  }

 private:
  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == pto_analysis::kSimdVfBlockName) {
      const SBlockNode *prev = current_;
      current_ = op;
      StmtExprVisitor::VisitStmt_(op);
      current_ = prev;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Mark(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Mark(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void Mark(const Buffer &buffer) {
    if (current_ == nullptr) {
      outside_.insert(buffer->data.get());
    } else {
      blocks_[buffer->data.get()].insert(current_);
    }
  }

  const SBlockNode *current_ = nullptr;
  std::unordered_set<const VarNode *> outside_;
  std::unordered_map<const VarNode *, std::unordered_set<const SBlockNode *>> blocks_;
};

class SimdVFRoundTripEliminator final : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    SimdVFRoundTripEliminator mutator;
    mutator.locality_.Scan(f->body);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return GetRef<PrimFunc>(fptr);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != pto_analysis::kSimdVfBlockName) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Stmt region = op->body;
    Stmt new_region = EliminateInRegion(region, op);
    if (new_region.same_as(region)) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Array<Buffer> alloc_buffers = PruneAllocBuffers(op, new_region);
    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, new_region,
                  op->init, alloc_buffers, op->match_buffers, op->annotations);
  }

 private:
  Stmt EliminateInRegion(const Stmt &region, const SBlockNode *block) {
    Stmt current = region;
    // Iteration budget: each rule deletes exactly one top-level statement
    // (the copy unit is erased and its neighbour replaced in place), so the
    // initial statement count is a correct upper bound on the number of
    // rounds — it never silently truncates a region with many round trips
    // (the previous fixed cap of 64 could). The budget only guards against
    // an unexpected non-shrinking rewrite.
    const int max_rounds = static_cast<int>(RegionStatements(region).size());
    for (int round = 0; round < max_rounds; ++round) {
      RegionModel model = BuildRegionModel(current);
      Stmt next;
      if (TryForwardRetarget(model, block, &next)) {
        current = next;
        continue;
      }
      if (TryBackwardCopyProp(model, block, &next)) {
        current = next;
        continue;
      }
      break;
    }
    return current;
  }

  bool DeclaredAndLocal(const UnitShape &unit, const SBlockNode *block,
                        const std::unordered_set<const VarNode *> &declared) const {
    if (!unit.exists || !unit.one_store || !unit.store_identity ||
        !unit.store_unpredicated) {
      return false;
    }
    const VarNode *t = unit.store->buffer->data.get();
    return declared.count(t) != 0 && locality_.LocalTo(t, block);
  }

  /*! True when every load recorded for the buffer belongs to one unit. The
   * consumer may read the round-trip buffer several times (e.g. a value used
   * both by the compare and by the value arm); they all move to the source
   * buffer together. */
  static bool AllLoadsIn(const BufferAccessRecord &rec, int ordinal) {
    for (int unit : rec.load_units) {
      if (unit != ordinal) {
        return false;
      }
    }
    return true;
  }

  std::unordered_set<const VarNode *> DeclaredBuffers(const SBlockNode *block) const {
    std::unordered_set<const VarNode *> declared;
    for (const Buffer &buffer : block->alloc_buffers) {
      declared.insert(buffer->data.get());
    }
    return declared;
  }

  /*! The shared guard of the two rules over one adjacent producer /
   * consumer pair: the producer's buffer T is written by exactly one
   * unpredicated identity store (the producer itself, on a declared
   * region-local buffer), both units are 1D and divisible with equal
   * extents, and T is read only by the consumer (identity loads, no
   * other use, nothing predicated). Returns T's data var when every
   * shared guard passes, nullptr otherwise; the rule-specific checks
   * (each side's copy shape and the OUT/S records) stay with the
   * callers. */
  const VarNode *MatchAdjacentTempPair(
      const UnitShape &producer, const UnitShape &consumer,
      const SBlockNode *block,
      const std::unordered_set<const VarNode *> &declared,
      const RegionModel &model) const {
    if (!DeclaredAndLocal(producer, block, declared)) {
      return nullptr;
    }
    if (!producer.one_d || !producer.divisible || !consumer.one_d ||
        !consumer.divisible || producer.extent != consumer.extent) {
      return nullptr;
    }
    const VarNode *t = producer.store->buffer->data.get();
    auto tit = model.access.find(t);
    if (tit == model.access.end()) {
      return nullptr;
    }
    const BufferAccessRecord &t_rec = tit->second;
    if (t_rec.predicated || t_rec.other != 0 || t_rec.loads < 1 ||
        !t_rec.loads_identity || t_rec.stores != 1 ||
        t_rec.store_units[0] != producer.ordinal ||
        !AllLoadsIn(t_rec, consumer.ordinal)) {
      return nullptr;
    }
    return t;
  }

  /*! Store forwarding: `T[i] = expr` + adjacent `OUT[i] = T[i]` ->
   * `OUT[i] = expr`. */
  bool TryForwardRetarget(const RegionModel &model, const SBlockNode *block,
                          Stmt *out) const {
    std::unordered_set<const VarNode *> declared = DeclaredBuffers(block);
    for (size_t pos = 0; pos + 1 < model.stmts.size(); ++pos) {
      const UnitShape *producer = UnitAt(model, pos);
      const UnitShape *consumer = UnitAt(model, pos + 1);
      if (producer == nullptr || consumer == nullptr) {
        continue;
      }
      // This rule's shape: the consumer is the pure copy `OUT[i] = T[i]`.
      if (!consumer->one_store || !consumer->store_identity ||
          !consumer->store_unpredicated || !consumer->pure_copy) {
        continue;
      }
      const VarNode *t =
          MatchAdjacentTempPair(*producer, *consumer, block, declared, model);
      if (t == nullptr) {
        continue;
      }
      if (consumer->copy_source->data.get() != t) {
        continue;
      }
      Buffer out_buffer = consumer->store->buffer;
      const VarNode *o = out_buffer->data.get();
      if (o == t || out_buffer->dtype != producer->store->buffer->dtype) {
        continue;
      }
      // `OUT` carries exactly the consumer's copy and nothing else in the
      // region: no other writer, no read. After the rewrite the producer's
      // store becomes the only write to it.
      auto oit = model.access.find(o);
      if (oit == model.access.end()) {
        continue;
      }
      const BufferAccessRecord &o_rec = oit->second;
      if (o_rec.predicated || o_rec.other != 0 || o_rec.loads != 0 ||
          o_rec.stores != 1 || o_rec.store_units[0] != consumer->ordinal) {
        continue;
      }
      std::vector<Stmt> stmts = model.stmts;
      stmts[pos] = StoreRetargeter(producer->store, out_buffer).Apply(stmts[pos]);
      stmts.erase(stmts.begin() + static_cast<std::ptrdiff_t>(pos + 1));
      *out = RebuildRegion(model.scope, stmts);
      return true;
    }
    return false;
  }

  /*! Copy propagation: `T[i] = S[i]` + adjacent consumer of `T[i]` ->
   * consumer reads `S[i]`. */
  bool TryBackwardCopyProp(const RegionModel &model, const SBlockNode *block,
                           Stmt *out) const {
    std::unordered_set<const VarNode *> declared = DeclaredBuffers(block);
    for (size_t pos = 0; pos + 1 < model.stmts.size(); ++pos) {
      const UnitShape *producer = UnitAt(model, pos);
      const UnitShape *consumer = UnitAt(model, pos + 1);
      if (producer == nullptr || consumer == nullptr) {
        continue;
      }
      // This rule's shape: the producer is the pure copy `T[i] = S[i]`.
      if (!producer->pure_copy) {
        continue;
      }
      const VarNode *t =
          MatchAdjacentTempPair(*producer, *consumer, block, declared, model);
      if (t == nullptr) {
        continue;
      }
      const VarNode *s = producer->copy_source->data.get();
      if (s == t || producer->copy_source->dtype != producer->store->buffer->dtype) {
        continue;
      }
      // Reading `S` at the consumer must observe the value the copy captured:
      // nothing may write `S` anywhere in the region, and adjacency leaves no
      // statement that could.
      auto sit = model.access.find(s);
      if (sit != model.access.end() && sit->second.stores != 0) {
        continue;
      }
      std::vector<Stmt> stmts = model.stmts;
      stmts[pos + 1] = LoadRetargeter(t, producer->copy_source).Apply(stmts[pos + 1]);
      stmts.erase(stmts.begin() + static_cast<std::ptrdiff_t>(pos));
      *out = RebuildRegion(model.scope, stmts);
      return true;
    }
    return false;
  }

  /*! The reserved PTO fragment backings must survive this pass: they carry
   * no accesses until VectorizeParallelToPTO emits them (their whole point
   * is to be reserved ahead of scheduling), so an access-based pruning
   * would drop the allocation that those accesses will reference. */
  std::unordered_set<const VarNode *> PlannedBackingVars(
      const SBlockNode *block) const {
    std::unordered_set<const VarNode *> vars;
    auto value = block->annotations.Get(kPtoSlotPlan);
    if (!value.has_value()) {
      return vars;
    }
    auto plans = value.value().try_cast<Array<PtoSlotPlan>>();
    if (!plans.has_value()) {
      return vars;
    }
    for (const PtoSlotPlan &plan : plans.value()) {
      vars.insert(plan->backing->data.get());
    }
    return vars;
  }

  /*! Drop block allocations that no longer take part in any access of the
   * rewritten region, except the planned slot backings above. */
  Array<Buffer> PruneAllocBuffers(const SBlockNode *block,
                                  const Stmt &region) const {
    std::unordered_set<const VarNode *> used;
    PostOrderVisit(region, [&used](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>()) {
        used.insert(load->buffer->data.get());
      } else if (const auto *store = node.as<BufferStoreNode>()) {
        used.insert(store->buffer->data.get());
      } else if (const auto *var = node.as<VarNode>()) {
        used.insert(var);
      }
    });
    for (const VarNode *var : PlannedBackingVars(block)) {
      used.insert(var);
    }
    Array<Buffer> kept;
    for (const Buffer &buffer : block->alloc_buffers) {
      if (used.count(buffer->data.get()) != 0) {
        kept.push_back(buffer);
      }
    }
    return kept.size() == block->alloc_buffers.size() ? block->alloc_buffers : kept;
  }

  RegionLocalityScan locality_;
};

}  // namespace

namespace transform {

tvm::transform::Pass EliminateSimDVFRoundTrip() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return SimdVFRoundTripEliminator::Substitute(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.EliminateSimDVFRoundTrip", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.EliminateSimDVFRoundTrip",
                        EliminateSimDVFRoundTrip);
}

}  // namespace transform

}  // namespace tl
}  // namespace tvm
