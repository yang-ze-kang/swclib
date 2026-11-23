# Fiber 参照的错误检测：schema 2

## 参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| distance_tolerance | 3 | 容差 r，也是 FiberMetric.dist_threshold |
| coverage_threshold | 0.9 | 近似覆盖阈值；保留覆盖长度达到最优值 τ 倍的参照候选 |
| short_length | None，即 2r | 短几何段参考长度，不过滤已证实的短拓扑错接 |
| min_error_length | 3 | 持续失配/缺失确认长度；明确跨分支转移不受此下限限制 |
| sample_step | 0.5 | 检测采样最大间距，保留原始节点与折线 |
| other_gts | None | `{明确神经元标识: Swc或路径}`，用于证实跨神经元错接 |
| fiber_metric_options | None | 现有 FiberMetric 的其他参数 |
| max_fiber_pairs | 4,000,000 | 精确匹配矩阵预算，超过显式报错，None 可解除 |
| strategy | fiber | legacy 仅用于复现旧持续失配计数 |

输入只读，不重新定根、不自动对齐、不改变父子关系。空预测或只有 seed 时保留立即终止。

## 原始评分独立保留

直接调用现有 `FiberMetric.run(..., return_fibers=True)`，不修改该类。
`fiber_metric` 保留原始 TP/FP/FN、precision/recall/f1、IoU 矩阵、匹配与 FN 索引。
使用显式 forest 输入，保持原始折线，resample_step=None；其他默认值来自现有类，
包括 iou_threshold=0.8、dist_sample=1、min_fiber_length=5。实际参数保存于
`fiber_metric_parameters`。比较其他评分时必须统一参数。
如需缩放或根对齐，应在调用前明确应用于双方；接口拒绝仅在评分内部改变几何。

FP/FN 用于解释和关联，同时检查 TP 所在树的局部连接，高 IoU 不保证没有错接。
短路径也可能被原评分的 min_fiber_length 排除，检测仍检查其真实树边。
分类不会改写评分：原始 FP/FN 可以被解释为几何偏差或重复预测，而不成为确认错误。

## 匹配、分类与去重

覆盖率按预测弧长积分，不按节点数。GT 参照按覆盖长度排序，近似并列保留候选。
完整父子路径的连续对应检查保留所有可行 GT 状态；不把最高覆盖、最近点或 Hungarian
配对直接当作因果证据。`fiber_references` 保存参照，`fiber_catalog` 保存 fiber 根、末端和长度。

同一条无分叉 GT 路径上局部失配后恢复，记录几何偏差，不直接计 FJ 或 break，也不证明
GT 标错。跨过 GT 分叉或存在多个不相容解释时列为待定。若确有非法跨 GT 分支的证据，
即使连接短于 2r 或后来又接回原路径，仍保留 FJ。

正确 GT 延续仍在、同时多出错误路径时属于 extra_branch；另一条合法 GT 子分支
不能替代当前应追踪分支。合法分支继续而另一子分支没有恢复，是漏分支。
末端过度延伸沿用已确认口径。缺乏 GT 支持的路径作为候选；只有提供明确身份的其他
神经元 GT，且异常后存在持续其他 GT 支持，才可确认 merge。孤立片段本身不证明两神经元相连。

GT 缺失分析同时核对预测连接。提前终止有可靠预测末端时定位于真实末端；断裂返回 GT
缺失区间及相关片段端点。远端片段即使与近端空间重叠，也不能代替实际父子连接。
已有另一条预测完整继续时，额外短路径可解释为重复预测，避免重复报 break。
没有可靠预测末端或明确错接原因的缺失后续列为待定，不一律叫提前终止。

共享主干只生成一次事件，再关联多个 fiber。只有结构上唯一对应的已确认 FJ 才抑制
同处缺失，保存在 `suppressed_break_events`；候选原因用 `associated_candidate_ids` 关联。
不按空间距离单独合并事件。`fiber_explanations` 保存每个 FP/FN 的首个主要解释和全部
相关事件；证据不足时 status=undetermined，不强行选择。

## 输出分组

| 分组 | error_type |
| --- | --- |
| 已确认 FJ | wrong_connection、extra_branch、cross_neuron_merge、terminal_overextension |
| 已确认 break | premature_termination、disconnection、missed_branch、missing_subtree |
| 候选 / 待定 | false_branch_candidate、false_fragment_candidate、unsupported_path_candidate、ambiguous_connection、ambiguous_missing_branch、uncertain_missing_continuation |
| 诊断 | geometric_deviation、pure_reversal、duplicate_tracing、short_geometry |

`events` 包含全部分组。`false_junction_events` / `break_events` 只含已确认事件，
另有 `candidate_events` / `diagnostic_events`。每个事件含 id、status、位置或区间、
相关预测/GT fiber、原始节点/边和 evidence。

`counts` 只统计已确认事件；`first_fj_counts` 筛选首次已确认 FJ，并保留全部独立 break。
`candidate_counts` / `diagnostic_counts` 分开统计。
`counts_with_candidates.false_junction_including_candidates` 是已确认 FJ 加潜在 FJ 候选，
不包含假片段和 FN 原因待定，后者单列。`is_first_fj` 与包含候选的 `is_first_anomaly`
分别表达首次确认 FJ 和首次预测异常，不能互换。

## 起点、确认点与容差内起偏

- start / matching_failure：首次持续匹配失效点，保留原始边 ID 与边内比例。
- location_interval：最后对应点到第一个异常点；不强行移动到分叉点。
- confirmation：达到持续证据要求的点；短拓扑转移可能与 start 相同。
- onset_review：原始边组成的上游待核对范围。
- onset_estimate：最初追偏估计，可能早于超过 r 的位置；证据不足时不输出单点。
- affected_subtree.node_ids：包含后来接回 GT 的全部预测后代，便于复核，不代表每个后代都错。

起偏估计建立 GT 对应路径上的稳定偏移基线，拟合“前段稳定、后段漂移”的变化点；
再检查相对基线的连续方向变化，必要时进一步回溯。恒定平行偏移不会自动成为起偏。
回溯计算窗口为 max(20r, 10×short_length)，截断时明确标注；原始复核范围可更长。
GT 参照有歧义或无稳定前段时保留区间。转向恰好位于原始顶点时，关联**向外走偏的边**，
避免把到达转向点的正确边归为最早错误预测。

目前是待人工校准的启发式：噪声以 MAD 估计，距离噪声下限为 0.1×sample_step，
趋势需超过三倍噪声并改善分段拟合；方向相对基线噪声判断。
结果保留 calibrated=False、exact_onset=None；起偏范围不是统计置信区间，
也不保证覆盖 SWC 中未记录的真实运动细节。root_to_start_length 保持阈值起点含义，
另存根到估计起点长度及范围。预测概率只来自记录，不解释为出错概率。

## 大 SWC 的实现

复用 swclib 的 SWC/Forest、重采样和 FiberMetric。子节点先挂接的 forest 构造避免
长链重复祖先检查。树边只采样一次；KD 邻域分块查询，按邻居总量自动缩小块。
长度覆盖使用 GT 末端区间累加，预测 DFS 进入/退出时复用共享主干贡献。
正确延续证据用范围索引，避免在长主干每个点复制所有下游叶子集合。
FN 归因使用结构邻接；日志重建不再每步复制整个节点字典。

原始 FiberMetric 仍有 G×P 矩阵、Hungarian 分配与 root-to-leaf 路径展开成本；
这是精确保留原评分的限制。大量分支加长主干的开销不能只按节点数预测。
输出全部后代节点也有与输出规模相关的成本。超过匹配矩阵预算会显式报错，
不默默近似、丢弃评分或强行把歧义归为正确。

## 验证与旧版

合成测试覆盖同路折弯、短错接、合法分支/漏分支、断开与重叠片段、重复预测、
歧义、merge 证据、容差内起偏、跨步概率，并直接对照原 FiberMetric 的结果。
旧测试显式调用 legacy；旧说明保留于 LEGACY_DETECTION.md。
新旧计数语义不同，不覆盖旧论文汇总；真实结果和验证脚本保存在用户指定 rebuttal 目录。
