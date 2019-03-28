import argparse
import os
import glob
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input_folder')
parser.add_argument('output_folder')
args = parser.parse_args()

def mkdir(path):
    if not os.path.exists(path):
        print('mkdir %s' % path)
        os.makedirs(path)

folder_name = os.path.basename(args.input_folder.rstrip('/'))
output_folder = os.path.join(args.output_folder, folder_name)
mkdir(output_folder)

attr_list = os.listdir(args.input_folder)
attr_path = [os.path.join(args.input_folder, tmp) for tmp in attr_list]
image_list = os.listdir(attr_path[0])

for image_name in tqdm(image_list):
    path_now = os.path.join(output_folder, image_name)
    mkdir(path_now)
    for attr in attr_list:
        src_image_path = os.path.join(args.input_folder, attr, image_name)
        dst_image_path = path_now+'/{}.jpg'.format(attr)
        shutil.copy(src_image_path, dst_image_path)
