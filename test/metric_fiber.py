from swclib.metrics.fiber_metric import FiberMetric

if __name__ == "__main__":
    metric = FiberMetric(dist_threshold=5, scale=(1, 1, 1 / 0.35), iou_threshold=0.8)
    # path1 = "data/swcs/output_r0.1.swc"
    # path1 = "data/swcs/output_mask.swc"
    path1 = "data/swcs/output_mask_r1.swc"
    # path2 = "/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk/annotation_idx_3_6539_1105_7627.swc"
    path2 = "data/swcs/anno3_refined_soma.swc"
    res = metric.run(path2, path1)
    print(res)
    res = metric.run(path1, path2)
    print(res)