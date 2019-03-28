'''
This script will collect the visualization of the network.

1. the correlation between different layers
2.
'''
import torch
import numpy as np
from .logger import logger
from tqdm import tqdm
from .util import toVariable

log = logger()
from .util import toTensor
import imageio
import cv2


def layer_correlation(layer1, layer2):
    '''
    compute the channel-wise correlation of two layers

    layer 1 and layer 2 are tensors, and located on CPU
    nBatch x nChannel x Height x Width


    return nBatch x nChannel-1 x nChannel-2
    '''
    shape1 = layer1.size()
    shape2 = layer2.size()
    log(shape1)
    log(shape2)

    feat1 = layer1.view(shape1[0], shape1[1] * shape1[2])
    feat2 = layer2.view(shape2[0], shape2[1] * shape2[2])
    m12 = feat1.mm(feat2.transpose(0, 1))
    log(m12)
    m1 = torch.diag(1 / torch.sqrt(torch.diag(feat1.mm(feat1.transpose(0, 1)))))
    log(m1)
    m2 = torch.diag(1 / torch.sqrt(torch.diag(feat2.mm(feat2.transpose(0, 1)))))
    log(m2)
    corr = m1.mm(m12).mm(m2)
    return corr


def writeTensor(save_path, tensor, nRow=16, row_first=False):
    '''
    use imageio to write the tensor
    :param tensor: nImage x 3 or 1 x height x width
    :param save_path: save path
    '''
    tensor = toTensor(tensor)
    nSample = tensor.size()[0]
    nCol = np.int16(nSample / nRow)
    all = []
    k = 0
    for iCol in range(nCol):
        all_ = []
        for iRow in range(nRow):
            now = tensor[k, :, :, :]
            now = now.permute(1, 2, 0)
            all_ += [now]
            k += 1
        if not row_first:
            all += [torch.cat(all_, dim=0)]
        else:
            all += [torch.cat(all_, dim=1)]
    if not row_first:
        all = torch.cat(all, dim=1)
    else:
        all = torch.cat(all, dim=0)
    all = all.cpu().numpy().astype(np.uint8)
    print('saving tensor to %s' % save_path)
    imageio.imwrite(save_path, all)


def untransformTensor(vggImageTensor):
    '''
    untransform the tensor that is pre-normalized to fit the VGG network
    :param vggImageTensor: nImage x 3 x height x width, it should be a tensor
    :return:
    '''
    vggImageTensor = vggImageTensor.cpu()
    mean = torch.Tensor((0.485, 0.456, 0.406))
    stdv = torch.Tensor((0.229, 0.224, 0.225))
    vggImageTensor *= stdv.view(1, 3, 1, 1).expand_as(vggImageTensor)
    vggImageTensor += mean.view(1, 3, 1, 1).expand_as(vggImageTensor)
    vggImageTensor = vggImageTensor.numpy()
    vggImageTensor[vggImageTensor > 1.] = 1.
    vggImageTensor[vggImageTensor < 0.] = 0.
    vggImageTensor = vggImageTensor * 255
    return vggImageTensor




def untransformVariable(vggImageVariable):
    mean = torch.Tensor((0.485, 0.456, 0.406))
    stdv = torch.Tensor((0.229, 0.224, 0.225))
    mean = toVariable(mean).cuda()
    stdv = toVariable(stdv).cuda()
    vggImageVariable *= stdv.view(1, 3, 1, 1).expand_as(vggImageVariable)
    vggImageVariable += mean.view(1, 3, 1, 1).expand_as(vggImageVariable)
    vggImageVariable[vggImageVariable.data > 1.] = 1.
    vggImageVariable[vggImageVariable.data < 0.] = 0.
    return vggImageVariable
