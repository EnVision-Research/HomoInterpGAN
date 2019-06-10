import os
import cv2
import numpy as np
import dlib
import argparse
import glob
from tqdm import tqdm
from . import util

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('util/shape_predictor_68_face_landmarks.dat')


def if_face_left(landmark, width):
    '''
    get the orientation of the image.
    :param landmark: the landmark detected by dlib
    :param width: the width of the image
    :return: True: the face is on the left side; False: the face is on the right side.
    '''
    left_count = 0

    for i in range(68):
        if int(landmark.part(i).x) < int(width / 2):
            left_count += 1
    if left_count < 32:
        return True
    else:
        return False

def get_orientation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    # print(rects)
    if len(rects) != 1:
        raise RuntimeError('The image is supposed to contain 1 face. {} face are detected.'.format(len(rects)))
    landmark = predictor(gray, rects[0])
    return if_face_left(landmark, img.shape[1])

def normalize_orientation(img):
    if not get_orientation(img):
        img = cv2.flip(img, 1)
    return img


def normalize_orientation_folder(folder_path, save_path):
    img_list = glob.glob(folder_path + '/*.jpg')
    util.mkdir(save_path)
    for img_path in img_list:
        img = cv2.imread(img_path)
        try:
            img = normalize_orientation(img)
            img_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(save_path, img_name), img)
        except RuntimeError:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='image root')
    # parser.add_argument('-sp', '--save_path', default=None, help='save path')
    parser.add_argument('save_path', help='the target save path')
    args = parser.parse_args()
    normalize_orientation_folder(args.folder, args.save_path)
