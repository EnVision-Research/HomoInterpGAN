import torch
import numpy as np
from .logger import logger
from tqdm import tqdm
from .util import toVariable

log = logger()
from .util import toTensor
import imageio
import cv2



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
