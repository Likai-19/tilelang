# PTO-AS PR #293: TileLang VMI load/store 修改方案

## 背景

PTO-AS PR #293 移除了 `pto.vmi.vload` 和 `pto.vmi.vstore` 的 `dist_mode`。TileLang 当前仍将它作为前端参数和 TIR annotation，并据此改变 load 结果形式、store 参数布局和 descriptor 覆盖范围。适配时应删除这层旧模式表达，让 TileLang 发出的 load/store 与 PTO-AS 当前接口一致。

本方案依据 TileLang 工作树 `feat/pto-parallel-vectorize`（编写时 HEAD `4f60f04`）及本机 PTO-AS 工作树中的 VMI 文档和 PR #293 实现。改动范围限于 TileLang PTO/VMI 前端、descriptor pass、PTO codegen、相关 verifier 和现有 PTO/VMI 测试。

## 目标接口

`T.vmi.vload` 和 `T.vmi.vstore` 不再接受、校验或写出 `dist_mode`。其余仍有效的访存参数保留：

```python
T.vmi.vload(source, offset=None, *, size, stride=None,
            block_stride=None, group=None)
T.vmi.vstore(values, destination, offset=None, mask=None, *,
             stride=None, block_stride=None, group=None)
```

实际 Python API 中已有的 `loc`、`ip` 参数保持原样。`vload` 产生一个结果；`vstore` 消费一个值，但保留现有 Python 参数名 `values`，使 `T.vmi.vstore(values=x, destination=...)` 关键字调用继续有效。这里的“一个结果”不要求 TIR dtype 一定是 vector：`size=1` 时仍可能使用标量 dtype 表示单 lane 值。

`group`/`stride` 和 `block_stride` 仍参与访存地址模式，因此必须保留在 API、descriptor 计算和 codegen 中。`vbrc`、`vintlv`、`vdintlv` 等各自作为独立 VMI 指令存在；本次不把它们与 load/store 组合成新的 TileLang 用法，也不要求 descriptor pass 分析这些寄存器数据操作。产生 pair 的重排等操作继续按各自现有规则处理。

## 前端 API

修改 `tilelang/ascend/language/vmi.py`：

- 从 `vload`、`vstore` 签名和 `_call_vmi` annotation 中删除 `dist_mode`。
- 删除只用于验证 `dist_mode` 及其与 `group`、`stride`、`block_stride` 组合关系的逻辑；保留 `group` 必须配套 `stride`、且不能与 `block_stride` 并用等访存参数校验。
- `vload` 始终构造单结果调用，不再因 `dintlv` 返回 `VmiPair`。
- `vstore` 保留参数名 `values`，但始终接收一个值，不再展开 `(even, odd)` 并生成双值调用。保留单值、dtype、mask、FP4 lane 和地址检查；错误信息不再提及 `dist_mode`。
- 保留现有 FP4 限制，但移除其中仅因 `dist_mode` 引入的分支。group、block-stride 和连续访问等既有约束不因此放宽。

同步更新 `src/op/builtin.h` 中 VMI TIR 调用的 ABI 注释，删除 `dist_mode`，确保注释列出的 annotation 和当前前端一致。

## Descriptor pass

修改 `src/ascend/transform/legalize_pto_vmi_memory_access.cc`。该 pass 对每个 `vload` 或 `vstore` 分别从该指令的地址和访问参数构造 descriptor；不把多条指令的地址合并，也不推断寄存器重排操作的用途。

- 每条 `vload`/`vstore` 进入 descriptor 处理时，先检查是否存在旧 `dist_mode` annotation；只要存在（包括值为 `"continuous"`），立即诊断失败。该检查必须位于参数布局解释、访存范围计算和 descriptor 构造之前，避免先按连续访问解释过时 TIR。codegen 保留独立防护。
- 删除 `VmiModes` 中的 `dist_mode`、对应的模式值解析和组合校验；仅保留上述旧 annotation 存在性检查。继续解析会影响内存访问的 `group`、`stride`、`block_stride`。
- `vload` 的 lane 数从 `size` annotation 得到；`vstore` 的 lane 数从唯一 value operand 的 dtype 得到。移除双 value 的参数布局、pair dtype 一致性检查和双 value 参数重建。
- 连续 load 继续覆盖当前实现计算出的完整物理向量范围；group load 继续按 group/stride 计算各行范围和物理读取范围；block-stride load/store 继续按 block 数和 block 间距计算范围。
- 连续 store 按该条 store 的 value lane 数计算覆盖范围；group 和 block-stride store 保留各自的范围计算。mask 不缩小 descriptor 声明的 store 范围。
- 删除 `brc` 将 load 范围缩为一个元素、`dintlv` 将访问范围扩大为两倍 lane 数的分支。广播或重排指令本身不改变 `vload` / `vstore` 的内存范围证明。
- 保留现有显式 `access_ptr` 验证、负 stride 处理、FP4 物理单位转换及溢出检查。

地址参数位置随单 value ABI 固定：`vload` 地址仍为第一个 TIR 参数，`vstore` 地址为第二个 TIR 参数（value、pointer、instruction offset、可选 mask）。相关诊断和任何从 `dist_mode` 推导位置的辅助逻辑都应同步删除。

## PTO codegen 与 legality verifier

修改 `src/ascend/codegen/codegen_pto.cc`：

- `ClassifyVmiCallResult_` 将 `vload` 一律分类为 VMI data，不再读取 `dist_mode` 决定其是否为 pair。
- `vstore` 的地址检查固定使用第二个 TIR 参数；移除双 value tuple 的特殊打印分支。
- 删除把 store 的 `dist_mode="dintlv"` 输出改成 PTO-AS `dist_mode="intlv"` 的特殊映射。生成 VMI load/store 的 PTODSL 时，若 TIR 仍带有旧 `dist_mode` annotation，codegen 应报错并确保不把它写入 PTODSL 参数；不得静默忽略该 annotation，以免改变过时 TIR 的访存语义。
- 保留 `vintlv`、`vdintlv`、`vunzip`、`vmull` 等其他 pair producer 的分类与 `pair_get` 处理。此次只取消 load/store 的旧 pair 语义。

检查并更新 `src/ascend/transform/verify_pto_codegen_legality.cc` 中 `MatchStore` 对 `dist_mode` 的特殊处理。删除对 `continuous` 的兼容接受；带有旧 `dist_mode` annotation 的调用仍不应被当作普通 masked store 匹配，避免把手工构造的过时 TIR 纳入该优化。普通 masked store 按 value、pointer、offset、mask 结构和其他现有条件识别。不为 descriptor pass、codegen 对旧 annotation 的诊断或这条 matcher 防护新增专门测试；必要的本地探针保存在工作树外。

## 测试范围

只调整当前 PR 已有测试，不新增专门验证旧 `dist_mode` 不再接受、`vload` 不是 pair、或 `vstore` 不接受 pair 的测试。也不为实现过程中的临时问题增加测试。

- 在仍表达有效访存行为的现有测试中删除 `dist_mode` 参数，按新的 descriptor 规则调整原有期望值。保留连续访问物理范围、group/stride、block-stride、显式 descriptor、负地址位移、FP4、mask 不缩小 store 范围等已有覆盖。
- 删除或收缩只验证已取消语义的旧断言，包括 `brc` load 单元素 descriptor、`dintlv` load 双结果范围、双输入 `vstore` 和 `dist_mode` 输出映射。不要用新的 `vbrc`/重排组合用例替换它们。
- 清理 `test_pto_masked_store_alignment.py` 中为 `continuous`/`dintlv` 构造的旧 annotation 输入；保留对普通 masked store、group 等仍相关 matcher 边界的现有覆盖。
- 清理 `test_pto_vmi_memory_access_run.py` 中依赖 `dist_mode="brc"` 的旧运行路径和输出断言；只运行并维护 PR 中已有且仍符合新接口的 VMI codegen/运行用例。
- `test_removed_vmi_parameters_fail_in_legal_scope` 中旧 `dist_mode` 值的拒绝断言不应转化为“新参数必须报错”的测试。删除这一过时 case；不新增替代的拒绝测试。

验证以现有相关 PTO 测试为准，重点检查 descriptor 结构、生成的 PTODSL、masked-store legality、VMI value-provenance，以及仍适用的 VMI codegen/运行用例。value-provenance 测试用于确认 `vload` 改为单结果分类后，其他现有 pair producer 仍按原规则工作。测试不覆盖 mask 对 interleave 类操作的运行时语义；该语义属于指令定义或 PTO-AS 降低，不由 descriptor 范围分析推断。本次不声明新写法与旧 interleave store 在部分 mask 下等价。

## 修改顺序

1. 更新 Python 前端签名及 TIR ABI 注释，保留 `vstore` 的 `values` 参数名，确认新调用不再生成 `dist_mode`。
2. 在 descriptor 处理入口拒绝旧 `dist_mode` annotation，再简化参数解析和访问范围分支，固定单值 store 参数布局。
3. 清理 PTO codegen 和 legality verifier 中依赖 `dist_mode` 的分类、地址索引、打印与匹配逻辑；codegen 遇到旧 load/store annotation 时诊断失败，不能将其输出到 PTODSL。
4. 按上述范围修订 PR 已有测试，移除旧模式专属断言，不增加中间过程用例。
5. 检查 TileLang 的 PTO/VMI 路径中是否还存在 load/store `dist_mode` 依赖，并运行保留下来的相关测试。

## 明确不做

- 不改 PTO-AS，也不实现 `vbrc`、`vintlv`、`vdintlv` 的组合策略。
- 不跨多条 memory op 合并 descriptor；每条 load/store 单独证明自己的访问范围。
- 不根据 mask 推断实际 store lane 子集，也不在 descriptor pass 中解释 interleave 或 deinterleave 的 mask 行为。
- 不增加对旧参数拒绝、单结果 load、pair store 拒绝的专门测试。
- 不改公共代码或 Ascend 非 PTO 后端逻辑。
