''' Visualization script

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/7/5

Required Inputs:
    --mesh_path (str): Path for mesh. The format is `<scannet path>/scans/<scene name>/<scene name>_vh_clean_2.ply`.
    --label_path (str): Path for labels. The format is `<result path>/<scene name>/epoch_<epoch>/<file name>.txt`.
    --label_type (str): Type of labels. ['instance', 'semantic', 'segment']
    					The type determines visualization colors. 
                    	For semantic labels, each color is specific to one class. 
                    	For instance and segment labels, the colors are randomly selected.

Important Optional Inputs:
    --shuffle (store_true): Whether to randomly shuffle colors in visualization. 
                    		If the color distribution in the visualization result is bad, you can shuffle the colors.

Example Usage: 
	python visualize.py --mesh_path <mesh path> --label_path <label path> --label_type semantic

'''

import argparse
from dataset.scannet.util import visualize_labels, visualize_grouping_process

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', type=str, required=True, 
                    help='Path for mesh. The format is `<scannet path>/scans/<scene name>/<scene name>_vh_clean_2.ply`.')
parser.add_argument('--label_path', type=str, required=True, 
                    help='Path for labels. The format is `<result path>/<scene name>/epoch_<epoch>/<file name>.txt`.')
parser.add_argument('--label_type', type=str, required=True, 
                    help='Type of labels. The type determines visualization colors. \
                    		For semantic labels, each color is specfic to a class. \
                    		For instance and segment labels, the colors are randomly selected.',
                    choices=['instance', 'semantic', 'segment'])
parser.add_argument('--shuffle', action='store_true',
                    help='Whether to randomly shuffle colors in visualization. \
                    		If the color distribution in the visualization result is bad, you can shuffle the colors.')
opt = parser.parse_args()

visualize_labels(args.mesh_path, args.label_path, label_type=args.label_type, shuffle=args.shuffle)

	