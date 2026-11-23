# SWC 错误检测（默认 Fiber 策略，schema 2）

```python
from swclib.error_detection import detect_errors, detect_trace_errors

result = detect_errors("pred.swc", "gt.swc")
print(result["fiber_metric"]["TP"], result["fiber_metric"]["FP"], result["fiber_metric"]["FN"])
print(result["counts"])                  # 已确认 FJ + break
print(result["first_fj_counts"])         # 首次已确认 FJ + 全部独立 break
print(result["candidate_counts"])        # 候选 / 待定单列
print(result["counts_with_candidates"]) # 含候选汇总；不是已确认 FJ
for event in result["events"]:
    print(event["status"], event["error_type"], event["start"]["xyz"])

recorded = detect_trace_errors("/data/run/trace_steps.jsonl", gt="gt.swc")
```

输入可以是 Swc 或路径，顺序是 pred、GT，只读且不修改输入拓扑。默认 r=3、τ=0.9、
short_length=2r、sample_step=0.5。原始 Fiber metric 不因错误分类而改变。
查看器同步显示已确认事件、候选/待定、几何诊断及容差内起偏估计。

- [完整策略、参数、类型、定位与性能说明](FIBER_STRATEGY.md)
- [追踪预测来源和概率](TRACE_DETECTION.md)
- [旧版说明](LEGACY_DETECTION.md)：仅适用于 strategy="legacy"，默认不再把持续偏离直接认定 FJ。

验证：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q swclib/error_detection/tests
python -m pytest -q swclib/visualization/tests/test_error_detection.py swclib/visualization/tests/test_trace_error_detection.py
```
