from swclib.metrics.keypoint_metric import KeypointMetric

if __name__ == "__main__":
    metric = KeypointMetric(threshold_dis=20, scale=(1, 1, 1 / 0.35))
    # path1 = "data/swcs/output_r0.1.swc"
    path1 = "data/swcs/output_mask_r1.swc"
    path2 = "data/swcs/anno3_refined_soma.swc"
    res = metric.run(path2, path1)
    print(res)
    # res = metric.run(path1, path2)
    # print(res)