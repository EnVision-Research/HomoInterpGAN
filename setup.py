from util import faceflip
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('folder', help='image root')
parser.add_argument('save_path', help='the target save path')
args = parser.parse_args()
print('normalizing face orientations')
faceflip.normalize_orientation_folder(args.folder, args.save_path)
