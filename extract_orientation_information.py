import pandas as pd
import os
import cv2
import glob
from util import faceflip
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-ic', '--input_csv')
parser.add_argument('-oc', '--output_csv')
parser.add_argument('--data_path')
args = parser.parse_args()

frame = pd.read_csv(args.input_csv)
img_list = list(frame['name'])

frame['orientation'] = 'unknown'
for img_name in tqdm(img_list):
    img_path = os.path.join(args.data_path, img_name)
    img = cv2.imread(img_path)
    try:
        orientation = faceflip.get_orientation(img)
    except RuntimeError:
        continue
    if orientation:
        frame.loc[frame.name == img_name, 'orientation'] = 'left'
    else:
        frame.loc[frame.name == img_name, 'orientation'] = 'right'
frame.to_csv(args.output_csv)
