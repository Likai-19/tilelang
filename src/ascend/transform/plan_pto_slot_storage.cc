// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file plan_pto_slot_storage.cc
 * \brief PlanPtoSlotStorage: declare the UB backing of every local.fragment
 * buffer used by a converting PTO SIMD_VF region, before scheduling.
 *
 * The pass is deliberately bookkeeping-only: it reads the fragment layouts
 * from the SBlock layout_map annotations, computes each slot's entry count /
 * vector type / reserved bytes, appends a hidden shared.dyn backing buffer
 * and a PtoSlotPlan record to the region's SBlock, and changes nothing else.
 * The backing carries no reads or writes until Vectorize emits slot_alloc,
 * so the scheduler sees no phantom dependencies. Consumers: the AutoSchedule
 * capacity budget, Verify (plan consistency) and Vectorize (slot
 * construction).
 */

#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

#include "../../layout/layout.h"
#include "../../op/operator.h"
#include "../../op/utils.h"
#include "../target_utils.h"
#include "parallel_to_pto_utils.h"
#include "pto_slot_plan.h"

namespace tvm {
namespace tl {

using namespace tirx;

PtoSlotPlanNode::PtoSlotPlanNode(Buffer fragment, Buffer backing,
                                 Var fragment_mem_var, int64_t count,
                                 int64_t vector_lanes,
                                 ffi::String elem_dtype_name,
                                 ffi::String vector_type_name, int64_t bytes)
    : fragment(std::move(fragment)), backing(std::move(backing)),
      fragment_mem_var(std::move(fragment_mem_var)), count(count),
      vector_lanes(vector_lanes), elem_dtype_name(std::move(elem_dtype_name)),
      vector_type_name(std::move(vector_type_name)), bytes(bytes) {}

PtoSlotPlan::PtoSlotPlan(Buffer fragment, Buffer backing, Var fragment_mem_var,
                         int64_t count, int64_t vector_lanes,
                         ffi::String elem_dtype_name,
                         ffi::String vector_type_name, int64_t bytes) {
  auto node = ffi::make_object<PtoSlotPlanNode>(
      std::move(fragment), std::move(backing), std::move(fragment_mem_var),
      count, vector_lanes, std::move(elem_dtype_name),
      std::move(vector_type_name), bytes);
  data_ = std::move(node);
}

namespace {

// Collect Array<PtoSlotPlan> values from SBlock annotations in document
// order. One implementation for every function-level consumer (capacity
// budget, alias exclusion, pass-internal dedup).
class PtoSlotPlanCollector : public StmtExprVisitor {
public:
  static std::vector<PtoSlotPlan> Run(const Stmt &body) {
    PtoSlotPlanCollector collector;
    collector(body);
    return std::move(collector.plans_);
  }

private:
  void VisitStmt_(const SBlockNode *op) final {
    auto value = op->annotations.Get(kPtoSlotPlan);
    if (value.has_value()) {
      auto plans = value.value().try_cast<ffi::Array<PtoSlotPlan>>();
      ICHECK(plans.has_value())
          << "Internal error: annotation " << kPtoSlotPlan
          << " must be an Array<PtoSlotPlan>, got " << value.value();
      for (const auto &plan : plans.value()) {
        plans_.push_back(plan);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::vector<PtoSlotPlan> plans_;
};

} // namespace

std::vector<PtoSlotPlan> CollectPtoSlotPlans(const Stmt &body) {
  return PtoSlotPlanCollector::Run(body);
}

std::set<const VarNode *> CollectPtoSlotBackingVars(const Stmt &body) {
  std::set<const VarNode *> vars;
  for (const auto &plan : CollectPtoSlotPlans(body)) {
    vars.insert(plan->backing->data.get());
  }
  return vars;
}

namespace {

// Storage Var of one declaration's backing address operand. The operand is
// the standard address expression the address passes rewrite: a
// tl.access_ptr over a BufferLoad of the backing before LowerAccessPtr, a
// tvm_access_ptr carrying the storage Var afterwards. Neither form is a
// data access.
const VarNode *DeclaredBackingVar(const PrimExpr &address) {
  const auto *call = address.as<CallNode>();
  if (call == nullptr) {
    return nullptr;
  }
  if (call->op.same_as(tl::access_ptr()) && !call->args.empty()) {
    if (const auto *load = call->args[0].as<BufferLoadNode>()) {
      return load->buffer->data.get();
    }
    return nullptr;
  }
  if (call->op.same_as(tirx::builtin::tvm_access_ptr()) &&
      call->args.size() >= 2U) {
    return call->args[1].as<VarNode>();
  }
  return nullptr;
}

class DeclaredBackingCollector : public StmtExprVisitor {
public:
  static std::set<const VarNode *> Run(const Stmt &body) {
    DeclaredBackingCollector collector;
    collector(body);
    return std::move(collector.vars_);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::vmi_declare_fragment_buffer()) &&
        !op->args.empty()) {
      if (const VarNode *var = DeclaredBackingVar(op->args[0])) {
        vars_.insert(var);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  std::set<const VarNode *> vars_;
};

} // namespace

std::set<const VarNode *> CollectDeclaredSlotBackingVars(const Stmt &body) {
  return DeclaredBackingCollector::Run(body);
}

std::set<const VarNode *> CollectAllSlotBackingVars(const Stmt &body) {
  std::set<const VarNode *> vars = CollectPtoSlotBackingVars(body);
  for (const VarNode *var : CollectDeclaredSlotBackingVars(body)) {
    vars.insert(var);
  }
  return vars;
}

namespace {

// TVM dtype name identical to the Python str(dtype) surface (vectorize /
// codegen keep their own copies; this one only feeds the plan's debug
// string field).
std::string SlotDtypeName(DataType dtype) {
  std::ostringstream os;
  if (dtype.is_bfloat16()) {
    return "bfloat16";
  }
  if (dtype.is_float8_e4m3fn()) {
    return "float8_e4m3fn";
  }
  if (dtype.is_float8_e5m2()) {
    return "float8_e5m2";
  }
  if (dtype.is_float()) {
    os << "float" << dtype.bits();
  } else if (dtype.is_int()) {
    os << "int" << dtype.bits();
  } else if (dtype.is_uint()) {
    os << "uint" << dtype.bits();
  } else if (dtype.is_bool()) {
    return "bool";
  } else {
    os << dtype;
  }
  return os.str();
}

class PlanPtoSlotStorageImpl : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    PlanPtoSlotStorageImpl mutator;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(fptr->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  // SBlocks in lexical scope, innermost last. Layout lookups walk the chain
  // so a plan may consume layouts written on an ancestor.
  std::vector<const SBlockNode *> block_stack_;
  // Fragment data Vars already planned in this function: a second region
  // touching the same fragment reuses the existing plan instead of
  // reserving a second backing (Verify rejects cross-region use later).
  std::set<const VarNode *> planned_fragments_;

  Stmt VisitStmt_(const SBlockNode *op) final {
    block_stack_.push_back(op);
    Stmt result = StmtExprMutator::VisitStmt_(op);
    block_stack_.pop_back();

    if (op->name_hint != "SIMD_VF") {
      return result;
    }
    const Stmt &region = op->body;
    pto::PtoRegionScan scan = pto::ScanPtoRegion(region);
    if (!scan.has_any_parallel) {
      // Pure hand-written VMI region: no conversion, no slots.
      return result;
    }
    return PlanRegion(op, Downcast<SBlock>(result));
  }

  Stmt PlanRegion(const SBlockNode *op, SBlock block) {
    const int64_t lanes = ReadSimdvfLanes(op);
    const auto accessed = CollectFragmentAccesses(op->body);

    std::vector<Buffer> new_backings;
    ffi::Array<PtoSlotPlan> new_plans;
    for (const Buffer &fragment : accessed) {
      if (planned_fragments_.count(fragment->data.get()) != 0) {
        continue;
      }
      const Fragment layout = FindFragmentLayout(fragment);
      if (!layout.defined()) {
        std::ostringstream oss;
        oss << "[PlanPtoSlotStorage] fragment buffer `" << fragment->name
            << "` is accessed inside a converting PTO SIMD_VF region but "
               "has no layout_map entry; the PTO layout inference must "
               "produce one for every accessed fragment buffer";
        throw pto::PtoAnalysisError(oss.str());
      }
      const int64_t count = EntryCountFromLayout(fragment, layout);
      const int64_t elem_bytes =
          (fragment->dtype.bits() + 7) / 8 * fragment->dtype.lanes();
      const int64_t raw_bytes = CheckedMul(count, lanes, fragment) * elem_bytes;
      // The alignment divides every supported element size, so the aligned
      // total stays a whole number of elements.
      const int64_t bytes = AlignUpSlotBytes(raw_bytes);
      ICHECK_EQ(bytes % elem_bytes, 0)
          << "Internal error: slot backing alignment is not a multiple of "
             "the element size for buffer "
          << fragment->name;

      std::ostringstream backing_name;
      backing_name << fragment->name << "_slot_backing";
      Buffer backing =
          decl_buffer({IntImm(DataType::Int(64), bytes / elem_bytes)},
                      fragment->dtype, backing_name.str(), "shared.dyn");
      // The handle Var is pre-created here and referenced by identity from
      // Vectorize, which inserts its sole definition (one
      // declare_fragment_buffer Bind per owning VF region). Creating it in
      // the plan keeps the handle stable across the scheduling passes
      // between the two transforms.
      Var fragment_mem_var(fragment->name + "_fragment_mem",
                           DataType::Handle());
      const std::string elem_name = SlotDtypeName(fragment->dtype);
      std::ostringstream vector_type;
      vector_type << elem_name << "x" << lanes;
      new_backings.push_back(backing);
      new_plans.push_back(PtoSlotPlan(fragment, backing, fragment_mem_var,
                                      count, lanes, elem_name,
                                      vector_type.str(), bytes));
      planned_fragments_.insert(fragment->data.get());
    }
    if (new_plans.empty()) {
      return block;
    }

    auto block_ptr = block.CopyOnWrite();
    for (const Buffer &backing : new_backings) {
      block_ptr->alloc_buffers.push_back(backing);
    }
    auto existing = block_ptr->annotations.Get(kPtoSlotPlan);
    ffi::Array<PtoSlotPlan> merged = std::move(new_plans);
    if (existing.has_value()) {
      auto prior = existing.value().try_cast<ffi::Array<PtoSlotPlan>>();
      ICHECK(prior.has_value()) << "Internal error: annotation " << kPtoSlotPlan
                                << " must be an Array<PtoSlotPlan>";
      for (const auto &plan : prior.value()) {
        merged.push_back(plan);
      }
    }
    block_ptr->annotations.Set(kPtoSlotPlan, merged);
    return block;
  }

  int64_t ReadSimdvfLanes(const SBlockNode *op) const {
    auto anno = op->annotations.Get("tl.simdvf_lanes");
    ICHECK(anno.has_value())
        << "[PlanPtoSlotStorage] internal error: converting SIMD_VF block "
           "without the tl.simdvf_lanes annotation";
    auto imm = anno.value().try_cast<IntImm>();
    ICHECK(imm.has_value())
        << "[PlanPtoSlotStorage] internal error: tl.simdvf_lanes must be an "
           "integer constant";
    return imm.value()->value;
  }

  static std::vector<Buffer> CollectFragmentAccesses(const Stmt &region) {
    class FragmentAccessCollector : public StmtExprVisitor {
    public:
      static std::vector<Buffer> Run(const Stmt &body) {
        FragmentAccessCollector collector;
        collector(body);
        return std::move(collector.buffers_);
      }

    private:
      void VisitExpr_(const BufferLoadNode *op) final {
        Record(op->buffer);
        StmtExprVisitor::VisitExpr_(op);
      }
      void VisitStmt_(const BufferStoreNode *op) final {
        Record(op->buffer);
        StmtExprVisitor::VisitStmt_(op);
      }
      void Record(const Buffer &buffer) {
        if (!IsFragmentBuffer(buffer)) {
          return;
        }
        for (const Buffer &seen : buffers_) {
          if (seen.same_as(buffer)) {
            return;
          }
        }
        buffers_.push_back(buffer);
      }
      std::vector<Buffer> buffers_;
    };
    return FragmentAccessCollector::Run(region);
  }

  Fragment FindFragmentLayout(const Buffer &fragment) const {
    for (auto it = block_stack_.rbegin(); it != block_stack_.rend(); ++it) {
      auto value = (*it)->annotations.Get(attr::kLayoutMap);
      if (!value.has_value()) {
        continue;
      }
      auto map = value.value().try_cast<LayoutMap>();
      if (!map.has_value()) {
        continue;
      }
      if (map.value().count(fragment) != 0) {
        auto frag = map.value()[fragment].as<Fragment>();
        ICHECK(frag.has_value())
            << "[PlanPtoSlotStorage] layout_map entry for fragment buffer `"
            << fragment->name << "` must be a tl.Fragment, got "
            << map.value()[fragment]->GetTypeKey();
        return frag.value();
      }
    }
    return Fragment();
  }

  static int64_t EntryCountFromLayout(const Buffer &fragment,
                                      const Fragment &layout) {
    // Canonical form validation: the layout must be a PTO-legal Fragment
    // (exactly one forward_index dimension, replicate == 1, constant
    // positive thread extent). Non-canonical layouts (floating fragments
    // seeded as FullyReplicated, or SIMT-oriented annotated layouts)
    // would produce wrong reservation sizes.
    if (layout->GetForwardIndex().size() != 1u) {
      std::ostringstream oss;
      oss << "[PlanPtoSlotStorage] fragment layout for buffer `"
          << fragment->name << "` must have exactly one entry dimension, got "
          << layout->GetForwardIndex().size()
          << "; the PTO solver produces one entry dimension for every "
             "region fragment (floating buffers included)";
      LOG(FATAL) << oss.str();
    }
    const int64_t *replicate = as_const_int(layout->ReplicateExtent());
    if (replicate == nullptr || *replicate != 1) {
      std::ostringstream oss;
      oss << "[PlanPtoSlotStorage] fragment layout for buffer `"
          << fragment->name
          << "` must have replicate_size == 1 (non-canonical layout: "
             "floating or SIMT-annotated)";
      LOG(FATAL) << oss.str();
    }
    const int64_t *thread_extent = as_const_int(layout->ThreadExtent());
    if (thread_extent == nullptr || *thread_extent <= 0) {
      std::ostringstream oss;
      oss << "[PlanPtoSlotStorage] fragment layout for buffer `"
          << fragment->name
          << "` must have a constant positive thread extent";
      LOG(FATAL) << oss.str();
    }
    const Array<PrimExpr> output_shape = layout->OutputShape();
    const int64_t *count = as_const_int(output_shape[0]);
    ICHECK(count != nullptr && *count > 0)
        << "[PlanPtoSlotStorage] fragment layout for buffer `" << fragment->name
        << "` must have a constant positive entry count, "
           "got "
        << output_shape[0];
    return *count;
  }

  static int64_t CheckedMul(int64_t count, int64_t lanes,
                            const Buffer &fragment) {
    ICHECK(count <= std::numeric_limits<int64_t>::max() / lanes)
        << "[PlanPtoSlotStorage] slot capacity for buffer `" << fragment->name
        << "` overflows (" << count << " entries x " << lanes << " lanes)";
    return count * lanes;
  }
};

} // namespace

namespace transform {

tvm::transform::Pass PlanPtoSlotStorage() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return PlanPtoSlotStorageImpl::Substitute(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                                  "tl.PlanPtoSlotStorage", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PlanPtoSlotStorage", PlanPtoSlotStorage);
}

} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() { PtoSlotPlanNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
