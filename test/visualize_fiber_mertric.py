from pathlib import Path

from swclib.metrics.fiber_metric import FiberMetric


if __name__=='__main__':
    save_dir = Path("data/visualize_fiber2")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metric = FiberMetric(dist_threshold=5, scale=(1, 1, 1 / 0.35), iou_threshold=0.8)
    # path1 = "/gpfs-flash/hulab/yangzekang/neuron/neuron-seg/output_r0.1.swc"
    path1 = "data/swcs/output_mask_r1.swc"
    # path2 = "/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk/annotation_idx_3_6539_1105_7627.swc"
    path2 = "data/swcs/anno3_refined_soma.swc"
    res = metric.run(path2, path1, return_fibers=True)
    print(res)
    matches = res['matches']
    matches = sorted(matches, key=lambda x:x[2])
    for i, match in enumerate(matches):
        filename1 = f"fiber{i+1}_iou{match[2]:.4}_gold.swc"
        filename2 = f"fiber{i+1}_iou{match[2]:.4}_pred.swc"
        with open(save_dir/filename1, 'w') as f:
            f.writelines(res["gold_fibers"][match[0]].to_str_list((1, 1, 0.35)))
        with open(save_dir/filename2, 'w') as f:
            f.writelines(res["pred_fibers"][match[1]].to_str_list((1, 1, 0.35)))
    for i, fid in enumerate(res['FN_fiber_ids']):
        filename = f"fiber_false_negative_{fid}_ious{res['ious'][fid].max()}.swc"
        with open(save_dir/filename, 'w') as f:
            f.writelines(res["gold_fibers"][fid].to_str_list((1, 1, 0.35)))
