# SWC 父子路径错误检测

```python
from swclib.error_detection import detect_errors

result = detect_errors("/data/pred.swc", "/data/gt.swc")
# 也接受 swclib.data.swc.Swc 对象；参数顺序是 pred、gt。
print(result["counts"])           # 全部 FJ + 独立 break 的事件数
print(result["first_fj_counts"])  # 首次 FJ + 同一组独立 break 的事件数
for event in result["events"]:
    print(event["id"], event["error_type"], event["start"]["xyz"])
```

可选参数为 `distance_tolerance=3.0`、`min_error_length=3.0`、`sample_step=0.5`。
输入必须使用相同的 XYZ 坐标系，长度单位与输入一致；体素坐标下就是 voxel。
函数不对齐、不重置根、不改变父子关系，也不修改输入对象或文件。坏数据、空 GT、
无效阈值会报错。允许空预测；无有效父子追踪长度的预测统一记 1 次从 seed 立即终止，
但非空的单点 GT 自比较不报错。

## 五种错误类型

| `error_type` | 含义 | 定位来源 |
| --- | --- | --- |
| `wrong_connection` | 应有的 GT 正确延续被错误路径替代 | pred |
| `extra_branch` | 同一 GT 路径的正确延续保留，同时多出错误分支 | pred |
| `terminal_overextension` | 到达 GT 末端后继续错误延伸 | pred |
| `premature_termination` | 追踪提前终止 | GT |
| `missed_branch` | 未追踪应有的 GT 分支 | GT |

前三项组成 `false_junction`，后两项组成 `break_error`；`total` 是两者总和。
这里统计错误**事件**，不统计错误样本比例。持续偏离 GT subtree 的路径也算 FJ，
无需证明其接到了另一条已标注神经纤维。

匹配沿预测和 GT 的父子方向推进，空间容差默认 3，连续失配或缺失弧长至少 3 才确认。
每条原始边细分到最多 0.5，保留原始节点和父子拓扑。纯回溯单列
`reversal_diagnostics`，不计 FJ；对应 GT 缺失仍按 break 规则判断。
同一处被 FJ 解释的缺失不重复计 break；记录在 `suppressed_break_events`。
break 以连通缺失区域的入口计数，不把其全部下游缺失分支逐条累加。

`is_first_fj=True` 表示该事件在预测祖先路径上没有更早的 FJ，独立分支可以各有一次首次 FJ。
`events` 始终返回全部 FJ 和未被抑制的 break；`first_fj_counts` 仅筛除后续 FJ，
break 的范围不随之改变。`false_junction_events`、`break_events`、各 `*_count`
等字段保留原研究实现的详细审计信息。

## 位置、确认路径和受影响子树

所有事件包含 `id`（本次检测内唯一）、`error_type`、`source`、`start`、`anchor`。
位置记录包括：

- `xyz`：精确到本次重采样网格的 XYZ 坐标。
- `node_id`：位置恰好落在原始节点时为节点 ID，否则为 `None`。
- `edge_parent_id`、`edge_child_id`、`edge_fraction`：位置所在的原始有向边及插值比例。
  根节点位置的这三个字段为 `None`；其他位置满足 `xyz=(1-t)*parent+t*child`。
- `dense_index`：本次参数下的内部重采样索引，不能当作原始 SWC ID。

FJ 还包含：

- `confirmation`：持续失配达到最短弧长时的确认点，**不用于替代起点**。
- `confirmation_path_xyz`：起点到确认点的连续路径；`confirmation_path_node_ids` 给出涉及的原始下游节点 ID。
- `affected_subtree`：`source="pred"`、`root_node_id`、`node_ids` 和 `start`。
  `node_ids` 包含起点所在边的下游端点及其全部预测后代；若起点落在原始节点上，则包含该节点及其后代。
  不包含同一分叉的其他兄弟路径；包含后来接回 GT 的后代，便于完整审查。
  这不表示每个后代都是独立错误事件。
- `root_to_start_length`：沿预测父子路径从根到该起点的弧长。

从 `affected_subtree` 生成子树显示时，入口边从 `start.xyz` 截断，而不是从原父节点开始；
不要把入口边上尚未失配的一段也涂成错误。后代节点本身可从输入 `Swc.nodes[id]` 查询。
若需要落盘为独立 SWC，应为插值起点另建虚拟根，不改变原始预测文件。

break 的 `start` 与 `anchor` 属于 GT，并提供 `missing_segments_xyz` 用于显示本次匹配实际遗漏的线段，
不将已被覆盖的远端节点全部涂为缺失。立即终止的 break 定位于 GT seed。

## 定位含义与验证范围

FJ `start` 是后来满足持续阈值的那段失配的**首个采样点**；`anchor` 是此前的匹配锚点。
这是容差下的操作性定位，不能直接解释成几何拐点或人工认定的生物学接错点。
例如在 `(20,0,0)` 转向并垂直离开直线 GT，容差 3、步长 0.5 时，
起点为 `(20,3.5,0)`，确认点为 `(20,6.5,0)`。GT 提前终止的定位同样受容差影响。

实现移植自 `tracer/paper-ICLR/rebuttal/1.false-junction_break_error` 的
`topology_metric.py`、`event_subtypes.py`；包运行时不依赖该研究目录。
本次封装保持原有检测/归因口径。原研究审计中“额外错误分支与独立终止是否被去重混淆”的
边界案例仍应结合 `suppressed_break_events` 审查；本次迁移不擅自改变其分类规则。
合成测试和旧结果一致性检查验证实现及定位口径，不等同于真实数据上的人工精确率/召回率。

## 测试

在 swclib 仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q swclib/error_detection/tests
python -m pytest -q swclib/visualization/tests/test_error_detection.py swclib/visualization/tests/test_trace_replay.py
```

查看器入口与操作见 `swclib/visualization/README.md` 的“追踪错误检测”。

## 追溯具体预测和概率

另提供 `detect_trace_errors(trace_path, gt=None, ...)`，从 direct v2 记录复原最后保留的追踪树，
将 FJ 起点、匹配锚点和确认点精确关联到生成边的预测、候选和原始概率。
break 仅给出相关预测，不赋予虚假的唯一概率。接口字段与使用方法见 [追踪来源说明](TRACE_DETECTION.md)。

## FJ 前段回溯区间（2026-11-12）

FJ 的 `start` 保留为原来的匹配失效点，并用 `start_role="matching_failure"` 明确其含义。
它不是已确定的最早追偏点。新增 `onset_review`，用于检查失效点前面仍可能处于容差内的路径。
这一扩展不改变 FJ/break 判定、五类事件数、首次 FJ 标记、原来的长度或后代子树字段。

回溯沿该事件自身的父链进行，在最近的上游预测分叉、预测根或前次对应关系重置（重新接回 GT）处停止。
不跨到兄弟路径，也不机械地只退一个节点、3 voxel 或一轮预测。
回溯范围是结构性复核区间，可能包含正确部分；左边界不保证是真实起点，也不排除更早已有偏离。
对无分叉的长路径，区间可以一直到根节点，这是保留不确定性的结果。

字段：

- `status`：`uncertain`；没有上游路径时为 `no_upstream_path`。
- `start` / `end`：回溯边界与原匹配失效点；`boundary_kind` 为 `prediction_fork`、`prediction_root` 或 `gt_reentry`。
- `exact_onset=None`、`probability=None`、`boundary_is_error_onset=False`、`earlier_onset_not_excluded=True`：不指定确定起点或错误概率。
- `length_voxel` / `path_xyz`：区间沿路径弧长和折线，精确保留原 SWC 边。
- `segments`：各原始有向边的 `edge_parent_id`、`edge_child_id`、`start_xyz`、`end_xyz`、`start_fraction`、`end_fraction`、`length_voxel`。
  若重入点/失效点位于边内，只包含对应的部分边；零长度边不作为追踪段返回。

查看器以琥珀色虚线显示区间，黄色 `mismatch` 表示原失配点，粉色 `confirmed` 表示持续失配确认点。
“查看回溯区间”显示区间全貌，“区间内父子边”可逐段点击定位。
