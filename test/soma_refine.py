from swclib.data.swc_tree import SwcTree


if __name__ == "__main__":
    swc = SwcTree("/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk/annotation_idx_3_6539_1105_7627.swc")
    swc.rescale((1, 1, 1/0.35))
    swc.refine_somas()
    swc.rescale((1, 1, 0.35))
    swc.save_to_file("data/swcs/anno3_refined_soma.swc")