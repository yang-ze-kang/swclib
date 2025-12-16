from tifffile import tifffile
import numpy as np

from swclib.image.mask2swc import Mask2Swc   

if __name__ == "__main__":
    mask = tifffile.imread(
        # "/gpfs-flash/hulab/yangzekang/neuron/neuron-seg/outputs/neuron-seg/dynunet_cldice0.3_iter10000-trail2/preds/train_dynunet_cldice0/guolab-etv133/annotation_idx_3_6539_1105_7627.tif"
        # "/gpfs-flash/hulab/yangzekang/neuron/dataset/guolab-etv133/mask_radius3/annotation_idx_3_6539_1105_7627_mask.tif"
        "data/images/test_mask.tif"
    )
    mask = (mask > 0).astype(np.uint8)

    mask2swc = Mask2Swc()
    swc_lines = mask2swc.run(mask, 
                            #  swc_path="data/swcs/output_r0.1.swc", 
                             swc_path="data/swcs/output_mask_r1.swc", 
                             soma_path="data/swcs/soma2.swc",
                             radius=0.1, verbos=True)