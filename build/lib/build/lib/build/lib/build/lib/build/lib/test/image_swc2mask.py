from swclib.image.swc2mask import Swc2Mask
from pathlib import Path

if __name__ == "__main__":
    # swc_file = "/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk/annotation_idx_3_6539_1105_7627.swc"
    swc_file = "data/swcs/anno3_refined_soma.swc"
    output_mask = "data/images/test_mask.tif"
    voxel_size = (0.5, 0.5, 1.0)  # Example voxel size in micrometers

    converter = Swc2Mask(shape=(300, 300, 300), scale=(1, 1, 1/0.35), radius=1)
    converter.run(swc_file, output_mask)
    print(f"Mask saved to {output_mask}")
