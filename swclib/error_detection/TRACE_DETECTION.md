> 2026-11-12：默认已更新为 Fiber 策略（schema 2）。原始 Fiber TP/FP/FN 独立保留；
> 候选不计入已确认 FJ。来源信息覆盖预测侧的确认事件、候选和诊断。
> 新增 trace_origin.estimated_onset，指向估计起偏边的实际生成预测，可能早于 start 所在预测。
> 起偏估计不等于确定因果。所有检测参数可通过 detect_trace_errors(..., **options) 传入；
> other_gts 与主 GT 一样按记录的 SWC-to-voxel 变换处理。
> 当前完整口径见 [FIBER_STRATEGY.md](FIBER_STRATEGY.md)，下文来源字段仍适用。

# 从追踪记录定位错误预测

```python
from swclib.error_detection import detect_trace_errors

result = detect_trace_errors('/data/run/trace_steps.jsonl')
# 或显式提供 GT；支持 Swc 对象或 SWC 文件路径。
# result = detect_trace_errors('/data/run', gt='/data/gt.swc')

for event in result['false_junction_events']:
    origin = event['trace_origin']['start']
    print(event['id'], event['start']['xyz'], origin['status'])
    if 'prediction_step' in origin:
        print(origin['prediction_step'], origin['candidates'])
```

输入支持当前 direct v2 格式的 `trace_steps.jsonl` 文件或其目录。同目录必须存在
`trace_initial.npz`；GT 未提供时读取 `trace_replay.json` 的 `ground_truth`。
旧版 hierarchical v1 的回放仍可用，但本函数不把局部记录的节点 ID 当作全脑最终节点来猜测归因，
目前遇到 v1 会明确报错。

## 检测范围和坐标

按顺序应用 `added_nodes`、`removed_nodes`，重建记录末尾保留的追踪树，然后复用
`detect_errors` 检测五类错误。已经回退删除的临时错误不计入。不使用最近的预测窗口
或最高分候选来猜测 FJ 来源。

这棵树是**未经过导出舍入、重新编号、去重及重采样的记录几何**，并非直接读取最终输出 SWC。
因此节点 ID 保留日志中的 ID（可以从 0 开始），不能直接当作导出 SWC 的 ID；
检测坐标或临界事件数也可能因导出后处理略有差异。模型初始化提供的树也在检测范围内，
其中的错误标记为 `initial_tree`，不冒充本次模型生成的错误。

默认阈值保持 `distance_tolerance=3`、`min_error_length=3`、`sample_step=0.5`，单位为全局体素 XYZ。
日志已是 `global_xyz_voxels`，不会再次缩放。显式或自动读取的 GT 按记录元数据中的
`swc_to_voxel_scale_xyz` 和 `swc_to_voxel_offset_xyz` 变换一次，行为与回放一致；
传入的 `Swc` 对象不会被修改。相对 GT 文件路径按记录目录解析。

记录未正常结束时，`trace.complete=False`，忽略未写完的最后一行，仅检查最后一个完整记录的状态。
此时 break 是当前追踪状态的缺失，不能当作完成追踪后的结论。检测期间日志、初始快照或元数据
发生变化会报错，请待写入完成后重试。

## 返回字段

保留 `detect_errors` 的 `counts`、`first_fj_counts`、`events`、各类事件及位置字段。
默认仍返回全部事件，并用 `is_first_fj` 标记没有祖先 FJ 的入口。
新增字段：

- `trace`：日志路径、记录是否完成、预测次数、初始/新增/删除/保留节点数、GT 路径与坐标变换、
  日志实际记录的候选阈值、FJ 来源分类统计。
- `trace_predictions`：以字符串化的 **0-based 预测序号**为键的关联预测记录表，包含记录中的 step、
  文件行号、seed、根节点 ID、被选候选及原始概率、实际生成边、候选路径、完整 `scores` 和阈值后候选。
  只返回本次错误涉及的记录，避免复制整个日志。
- `trace_geometry`：仅在 `include_geometry=True` 时返回复原 `pred_nodes` 和变换后的 `gt_nodes`，
  每项含 `id`、`parent`、`xyz`，供显示使用。

### FJ 的 `trace_origin`

`start`、`anchor`、`confirmation` 分别记录检测起点、此前匹配锚点、持续偏离确认点所在边的来源。
每个来源包含以下信息（初始树无预测字段）：

- `prediction_step`：JSONL 中原样保留的 step。
- `prediction_index`：该文件中第几条 prediction，**从 0 开始**；可直接用于对应文件的回放位置。
- `log_path`、`log_line`：日志文件和 **从 1 开始**的行号。
- `recorded_node_id`、`edge_xyz`：该边的记录下游节点 ID 和原始有向边坐标。
- `candidates`：其 `actual_edges_xyz` 精确包含该边的所有候选，含原始 `candidate_index`、`probability`、`stop_reason`。
  候选编号保持模型原始索引，通常从 0 开始。
- `probability`：仅当恰好一个候选对应该边时提供其概率，否则为 `None`。

`status` 分为 `matched`（唯一候选）、`multiple_candidates`（共同前缀由多个候选共享）、
`candidate_unrecorded`（生成步骤已知，但记录未提供候选边）、`initial_tree`（初始树）。
不会因为某个候选概率更高就在共享前缀时强行选择它。
`confirmation_prediction_indices` 还列出从起点到确认点经过的各预测序号。

起点仍采用此前定义的“持续失配的首个采样点”，不等于超出容差前最早出现几何偏差的位置。
例如起点边由第 1 次预测生成，后续直到第 2 次预测才达到最短 3 voxel 持续条件，
错误起点来源是第 1 次，确认来源是第 2 次，两者分别返回。

这里的 p 是模型对**候选路径的原始输出概率**，不是该候选出错的概率，也不是经过校准的错误风险。

### break 的 `trace_origin`

break 没有对应的错误生成边，因此 `probability` 始终为 `None`。
在 GT 匹配锚点附近，以及漏分支的 GT 分叉附近，用当前匹配容差查询记录中的 seed，
通过 `related_prediction_indices` 返回相关预测（包含停止尝试，以及在分叉处只延续其他方向的预测）。
这是空间关联，并非唯一因果归因；附近其他路径也可能出现候选。
有相关记录时为 `related_predictions_only`，无记录时为 `unresolved`。

查看其中各候选的 `stop_reason`、概率、完整分数及回放上下文来人工核对；
不把最高分、另一条保留分支的概率或未采用候选的分数强行当作漏分支概率。

## 查看器

“插件 → 追踪错误检测 → 输入方式”选择 **追踪过程记录 + GT**，输入 JSONL 文件或其目录。
GT 留空时从元数据读取。检测后点击错误，详情列出来源预测、候选编号、概率和停止原因。
FJ 起点及确认点来自不同预测时分别展示；初始树和无法唯一归因会明确注明。

点击 **回放此预测** 跳到对应步骤，相关候选在回放列表中加框；仍可用左右键继续检查。
点击错误列表可返回错误定位。默认“首次 FJ”筛选不改变来源追溯规则。

## 前段回溯及预测来源（2026-11-12）

`start` 对应“匹配首次失效”，生成该点所在边的预测不一定是最早导致追偏的预测。
现在每个 FJ 附带 `onset_review`，沿当前父链回溯到最近的预测分叉、根或前次接回 GT 的位置；
完整几何定义见 [检测接口的回溯区间说明](README.md#fj-前段回溯区间2026-11-12)。
该区间用于复核而非确定错误起点，FJ/break 判定及计数不变。

记录模式还会为 `onset_review.segments` 的每条边添加 `trace_origin`：
通过原始节点 ID 和实际有向边关联到生成它的预测和候选，沿用 `matched`、`multiple_candidates`、
`initial_tree`、`candidate_unrecorded` 状态。多个候选共享一条边时保留全部概率，不选最高分代替唯一来源。
区间可能包含多次预测，`prediction_indices` 列出其 0-based 索引，
`provenance_counts` 汇总各边的来源状态，对应详情也加入顶层 `trace_predictions`。
被回退删除的节点/预测边不作为最终区间的来源。

```python
for event in result['false_junction_events']:
    review = event['onset_review']
    print(event['id'], review['status'], review['start']['xyz'], review['end']['xyz'])
    for segment in review['segments']:
        source = segment['trace_origin']
        print(segment['edge_parent_id'], segment['edge_child_id'],
              source.get('prediction_step'), source.get('candidates'))
```

查看器中“失配点边”“确认点边”和“回溯区间 · 待核对”分别标明来源；只参与回溯的预测卡片默认折叠，
可展开查看候选概率并点击“回放此预测”。区间内父子边可以逐条点击定位。
所有显示的候选概率仍是记录中的模型输出，不是“这一段最早出错”的概率；也不认定上一轮预测必定出错。
