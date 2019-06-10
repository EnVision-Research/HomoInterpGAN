import pandas as pd
import os
import cv2
import glob
from util import faceflip
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-ic', '--input_csv')
parser.add_argument('-oc', '--output_csv')
parser.add_argument('--data_path1')
parser.add_argument('--data_path2')
args = parser.parse_args()

frame = pd.read_csv(args.input_csv)
img_list = glob.glob(args.data_path2+'/*.jpg')

frame['orientation'] = 'unknown'
for img_name in tqdm(img_list):
    img_path1 = os.path.join(args.data_path1, img_name)
    img_path2 = os.path.join(args.data_path2, img_name)
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if np.sum(np.abs(img1-img2)) > 1:
        frame.loc[frame.name == img_name] = 'left'
    else:
        frame.loc[frame.name == img_name] = 'right'


frame.to_csv(args.output_csv)
