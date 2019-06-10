#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function
import time

timestamp = int(round(time.time()))
import numpy
import argparse
import json
import os.path
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import errno
from facealign import alignface
from facealign import imageutils


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read(ipath, dtype=numpy.float32):
    '''
    Returns a H x W x 3 RGB image in the range of [0,1].
    '''
    img = PIL.Image.open(ipath)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return numpy.asarray(img, dtype=dtype) / 255.0


def write(opath, I, **kwargs):
    '''
    Given a H x W x 3 RGB image it is clipped to the range [0,1] and
    written to an 8-bit image file.
    '''
    img = PIL.Image.fromarray((I * 255).clip(0, 255).astype(numpy.uint8))
    ext = os.path.splitext(opath)[1]
    if ext == '.jpg' or ext == '.jpeg':
        quality = kwargs['quality'] if 'quality' in kwargs else 95
        img.save(opath, quality=quality, optimize=True)
    elif ext == '.png':
        img.save(opath)
    else:
        # I do not want to save unknown extensions because there is no
        # expectation that the default save options are reasonable.
        raise ValueError('Unknown image extension ({})'.format(ext))


def fit_landmarks_to_image(template, original, Xlm, face_d, face_p, landmarks=list(range(68))):
    '''
    Fit the submanifold to the template and take the top-K.

    Xlm is a N x 68 x 2 list of landmarks.
    '''
    MX = numpy.empty((len(Xlm), 2, 3), dtype=numpy.float64)
    nfail = 0
    for i in range(len(Xlm)):
        lm = Xlm[i]
        try:
            M, loss = alignface.fit_face_landmarks(Xlm[i], template, landmarks=landmarks, image_dims=original.shape[:2])
            MX[i] = M
        except alignface.FitError:
            MX[i] = 0
            nfail += 1
    if nfail > 1:
        print('fit submanifold, {} errors.'.format(nfail))
    return MX


def warped_image_feed(S, MP, image_dims, input_path, output_path):
    '''
    Given a list of file paths, warp matrices and a 2-tuple of (H, W),
    yields H x W x 3 images
    '''
    for i, x in enumerate(S):
        I = read(input_path + "/" + x)
        split = x.split("/")
        directory = output_path

        for j, y in enumerate(split):
            if j != len(split) - 1:
                directory = directory + "/" + y
        temp = numpy.asarray(alignface.warp_to_template(I, MP[i], image_dims=image_dims))

        if not os.path.exists(directory):
            mkdir_p(directory)
        print(output_path + "/" + x)
        write(output_path + "/" + x, temp)



def get_image_list(subdir):
    S = set(['.jpg', '.png', '.jpeg'])
    result = []

    def error_fn(e):
        raise e

    for dirpath, dirnames, filenames in os.walk(subdir, onerror=error_fn, followlinks=True):
        for x in filenames:
            if (os.path.splitext(x)[1]).lower() in S:
                result.append(os.path.join(dirpath, x)[len(subdir) + 1:])
    result.sort()
    return result
    # with open(opath,'w') as f:
    #   for x in result:
    #     print(x,file=f)


if __name__ == '__main__':
    # get image list
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('-template', default='facealign/000215.jpg')
    args = parser.parse_args()
    fit_landmark = list(range(17)) + list(range(28,46))
    input_path = args.input_path
    output_path = args.output_path
    template_path = args.template
    image_list = get_image_list(input_path)
    # face detector
    face_d, face_p = alignface.load_face_detector()

    # detect landmarks
    template, original = alignface.detect_landmarks(template_path, face_d, face_p)
    minimum_resolution = 30
    image_dims = original.shape[:2]

    # if min(image_dims) < 200:
    #     s = float(minimum_resolution) / min(image_dims)
    #     image_dims = (int(round(image_dims[0] * s)), int(round(image_dims[1] * s)))
    #     original = imageutils.resize(original, image_dims)

    # # load landmarks
    # data = numpy.load('/data/xyshen/ryli/datasets/TrainAB/attributes.npz')
    # filelist = tuple(
    #     [fname.replace('/data/xyshen/deepfeatinterp/images/facemodel/trainB/', '') for fname in data['filelist']])
    # print(len(filelist))
    # map_name = {k: i for i, k in enumerate(filelist)}
    # image_list = [fname.replace('/data/xyshen/deepfeatinterp/images/facemodel/trainB/', '') for fname in
    #               data['filelist'] if '/trainB/' in fname]
    # idx = [map_name[x] for x in image_list]
    # print(len(idx))
    # landmarks = data['landmarks']
    # landmarks_list = landmarks[idx]
    # data.close()

    print('-----fit_landmarks_to_image-----')
    import glob
    from tqdm import tqdm
    from facealign.alignface import FitError
    image_list = glob.glob(input_path + '/*.jpg')
    for ipath in tqdm(image_list):
        try:
            landmark, _= alignface.detect_landmarks(ipath,face_d, face_p)
            landmark_list = [landmark]
            # print(landmark_list)
            M = fit_landmarks_to_image(template, original, landmark_list, face_d, face_p)
            image_list = [os.path.basename(tmp) for tmp in image_list]
            warped_image_feed([os.path.basename(ipath)], M, image_dims, input_path, output_path)
        except FitError:
            print('FitError')
