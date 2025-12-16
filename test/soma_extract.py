from swclib.data.swc_tree import SwcTree
from swclib.data.swc_soma import save_somas_to_file


tree = SwcTree('/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk/annotation_idx_3_6539_1105_7627.swc')

tree.rescale((1,1,1/0.35))
somas = tree.get_somas()
save_somas_to_file(somas, "soma2.swc")