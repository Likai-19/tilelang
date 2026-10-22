/*!
 * \file pto_slot_plan.h
 * \brief The PTO slot storage planning record shared by PlanPtoSlotStorage,
 * the UB capacity budget, VerifyParallelToPTO and VectorizeParallelToPTO.
 *
 * A plan maps one local.fragment buffer of a converting PTO SIMD_VF region
 * to a slot object: count entries of vector<lanes x elem_dtype>, backed by
 * an exclusively-reserved shared.dyn buffer.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_PTO_SLOT_PLAN_H_
#define TVM_TL_ASCEND_TRANSFORM_PTO_SLOT_PLAN_H_

#include <tvm/ffi/object.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <set>
#include <vector>

#include "../../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;

/*! Annotation key on the SBlock that owns the fragment allocation; the
 * value is an ffi::Array<PtoSlotPlan>. Plans reference the actual Buffer
 * and Var objects, never names. */
constexpr const char *kPtoSlotPlan = "tl.pto_slot_plan";

/*! Slot backing capacity alignment in bytes. Must stay in sync with the
 * MergeUBAllocations align_bytes the Ascend pipeline passes down
 * (tilelang.ascend.pipeline._MERGE_UB_ALIGNMENT). */
constexpr int kPtoSlotAlignmentBytes = 32;

/*!
 * \brief Planning record for one fragment buffer's slot storage.
 *
 * Written by PlanPtoSlotStorage before scheduling; consumed by the
 * AutoSchedule capacity budget, VerifyParallelToPTO (consistency) and
 * VectorizeParallelToPTO (slot construction). The element dtype of the
 * vector type equals fragment->dtype; vector_type_name mirrors the full
 * entry vector type (elem dtype x lanes) for cross-checking against the
 * declaration Vectorize emits.
 */
class PtoSlotPlanNode : public ffi::Object {
public:
  /*! The local.fragment buffer in the input TIR. */
  Buffer fragment;
  /*! Hidden shared.dyn backing buffer reserved for this slot. */
  Buffer backing;
  /*! Pre-created fragment storage handle Var (Handle dtype). Vectorize
   * inserts its definition — one declare_fragment_buffer Bind per VF
   * region — inside the owning tl.simdvf_scope; the plan never binds it. */
  Var fragment_mem_var;
  /*! Entry count from the buffer's layout (never element_count / lanes). */
  int64_t count = 0;
  /*! Logical lane count VL of the owning SIMD_VF. */
  int64_t vector_lanes = 0;
  /*! TVM dtype name of the element type (== fragment->dtype). */
  ffi::String elem_dtype_name;
  /*! Full entry vector type name, "<elem dtype name>x<lanes>" (e.g.
   * "float32x128"); must match the dtype/lanes of every access the handle
   * anchors. */
  ffi::String vector_type_name;
  /*! align_up(count * lanes * elem_bytes, kPtoSlotAlignmentBytes). */
  int64_t bytes = 0;

  PtoSlotPlanNode() = default;
  PtoSlotPlanNode(Buffer fragment, Buffer backing, Var fragment_mem_var,
                  int64_t count, int64_t vector_lanes,
                  ffi::String elem_dtype_name, ffi::String vector_type_name,
                  int64_t bytes);

  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<PtoSlotPlanNode>()
        .def_ro("fragment", &PtoSlotPlanNode::fragment)
        .def_ro("backing", &PtoSlotPlanNode::backing)
        .def_ro("fragment_mem_var", &PtoSlotPlanNode::fragment_mem_var)
        .def_ro("count", &PtoSlotPlanNode::count)
        .def_ro("vector_lanes", &PtoSlotPlanNode::vector_lanes)
        .def_ro("elem_dtype_name", &PtoSlotPlanNode::elem_dtype_name)
        .def_ro("vector_type_name", &PtoSlotPlanNode::vector_type_name)
        .def_ro("bytes", &PtoSlotPlanNode::bytes);
  }

  TVM_FFI_DECLARE_OBJECT_INFO("tl.PtoSlotPlan", PtoSlotPlanNode, ffi::Object);
};

class PtoSlotPlan : public ffi::ObjectRef {
public:
  TVM_DLL PtoSlotPlan(Buffer fragment, Buffer backing, Var fragment_mem_var,
                      int64_t count, int64_t vector_lanes,
                      ffi::String elem_dtype_name, ffi::String vector_type_name,
                      int64_t bytes);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PtoSlotPlan, ffi::ObjectRef,
                                                PtoSlotPlanNode);
};

/*! \brief Align \p value up to the slot backing alignment. */
inline int64_t AlignUpSlotBytes(int64_t value) {
  const int64_t align = kPtoSlotAlignmentBytes;
  return ((value + align - 1) / align) * align;
}

/*! \brief Collect every PtoSlotPlan recorded on SBlock annotations under
 * \p body, in document order. Intended for function-level consumers
 * (capacity budget, alias exclusion) that only need the full set. */
std::vector<PtoSlotPlan> CollectPtoSlotPlans(const Stmt &body);

/*! \brief The storage Vars of every slot backing named by a
 * tl.vmi.declare_fragment_buffer call under \p body.
 *
 * Late identification: the plans are consumed when Vectorize converts a
 * fragment buffer, so passes that run afterwards (InferBufferAliases,
 * MergeUBAllocations) cannot rely on the annotation. The declaration's
 * backing operand is the surviving authoritative link — it accepts both
 * address forms (tl.access_ptr over a BufferLoad before LowerAccessPtr,
 * tvm_access_ptr with the storage Var afterwards). */
std::set<const VarNode *> CollectDeclaredSlotBackingVars(const Stmt &body);

/*! \brief The storage Vars of every slot backing recorded under \p body,
 * either through a surviving plan or through a declaration. Passes that
 * may run before or after Vectorize should use this union. */
std::set<const VarNode *> CollectAllSlotBackingVars(const Stmt &body);

/*! \brief The storage Vars of every slot backing recorded under \p body. */
std::set<const VarNode *> CollectPtoSlotBackingVars(const Stmt &body);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PTO_SLOT_PLAN_H_
