/*!
 * \file pto_indexed_memory.h
 * \brief Shared allocation-owner contract for PTO indexed memory operations.
 *
 * Implemented in legalize_pto_vmi_memory_access.cc alongside the descriptor
 * legalization helpers shared by automatic and hand-written VMI accesses.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_PTO_INDEXED_MEMORY_H_
#define TVM_TL_ASCEND_TRANSFORM_PTO_INDEXED_MEMORY_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>

#include <string>
#include <vector>

namespace tvm {
namespace tl {
namespace pto {

/*! The existing allocation described by an indexed access. Resolving an owner
 * does not allocate, copy or initialize storage. */
struct IndexedMemoryOwner {
  tirx::Buffer buffer;
  PrimExpr origin;
  PrimExpr extent;
};

/*! Resolve the unique active allocation and validate its UB scope, element
 * type, dense physical layout and stable capacity. Source may be a same-dtype
 * view. The environment contains lexical allocations active at this point,
 * never a function-wide collection of every buffer sharing the data Var. */
IndexedMemoryOwner
GetIndexedMemoryOwner(const tirx::Buffer &source,
                      const std::vector<tirx::Buffer> &active_allocations,
                      arith::Analyzer *analyzer, const std::string &context,
                      const Span &span = Span());

/*! Check scalar instruction base D in owner-relative element units. The caller
 * separately preserves F = D + offsets for active accesses. */
void CheckIndexedMemoryBase(const IndexedMemoryOwner &owner,
                            const PrimExpr &instruction_offset,
                            arith::Analyzer *analyzer,
                            const std::string &context,
                            const Span &span = Span());

/*! Build a whole-owner may-access descriptor. Scatter does not assert dense or
 * unconditional write coverage through this descriptor. */
PrimExpr MakeIndexedMemoryDescriptor(const IndexedMemoryOwner &owner,
                                     bool is_write);

} // namespace pto
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PTO_INDEXED_MEMORY_H_
