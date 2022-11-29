import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import time
from functools import reduce
from sklearn import metrics

from loss import euclidean_dist


def time_now():
    return time.strftime('%y-%m-%d %H:%M:%S', time.localtime())

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_ir_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_ir_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_ir_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)
    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_ir_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_ir_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        
        N = np.maximum(len(train_color_label), len(train_ir_label))
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class IdentitySamplerUnbalanced(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_ir_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_ir_label, color_pos, thermal_pos, num_pos, batchSize, ir_ids=None, color_ids=None):
        uni_label = np.unique(train_color_label)
        if ir_ids is None :
            ir_ids = uni_label
        if color_ids is None:
            color_ids = ir_ids


        uni_label = np.unique(np.append(color_ids, ir_ids))

        self.n_classes = len(uni_label)

        N1 = reduce(lambda s, a: s + len(a), [color_pos[i] for i in color_ids], 0)
        N2 = reduce(lambda s, a: s + len(a), [thermal_pos[i] for i in ir_ids], 0)

        N = np.maximum(N1, N2)
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                if batch_idx[i] in color_ids:
                    sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                else:
                    sample_color = np.asarray([-1] * num_pos)

                if batch_idx[i] in ir_ids:
                    sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                else:
                    sample_thermal = np.asarray([-1]*num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def next_IDs(model, n, allIDs, currentIDs, trainset, color_pos, thermal_pos, transform_test):
    """ creates next intermediate domain by finding IDS with lower distance

    Returns:
        list of IDs
    """
    # return allIDs
    model.eval()
    availabeIDS = np.setdiff1d(allIDs, currentIDs)
    if len(availabeIDS) <= n:
        return availabeIDS
    # randomIDs = np.random.choice(availabeIDS, n, replace=False)
    dis={}
    with torch.no_grad():
        for id in availabeIDS:
            # f_c = torch.empty(len(color_pos[id]), model.pool_dim)
            # t_c = torch.empty(len(thermal_pos[id]), model.pool_dim)
            input_c = torch.stack([transform_test(trainset.train_color_image[i]) for i in color_pos[id]])
            input_t = torch.stack([transform_test(trainset.train_ir_image[i]) for i in thermal_pos[id]])
            feat_c = model(input_c.cuda(), None, modal=1)[0]
            feat_t = model(None, input_t.cuda(), modal=2)[0]
            #dis[id] = torch.linalg.norm(feat_c.mean(dim=0) - feat_t.mean(dim=0)).item()
            dis[id] = mmd_rbf(feat_c.detach().cpu().numpy(), feat_t.detach().cpu().numpy())
            print("dist {} is {:.4f}".format(id, dis[id]))

    sortedIDs = np.asarray([i[0] for i in sorted(dis.items(), key = lambda kv: kv[1])])
    return sortedIDs[:n]

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def getHardIndices(global_feat, labels):
    """
    Args:
        global_feat : a tensor [batch, feature_dim] features
        labels      : a tensor [batch] labels of features
    """

    dist_mat = euclidean_dist(global_feat, global_feat)

    N = dist_mat.size(0)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    ind = (labels.new().resize_as_(labels)
           .copy_(torch.arange(0, N).long())
           .unsqueeze(0).expand(N, N))

    p_inds = torch.gather(
        ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    n_inds = torch.gather(
        ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)
    return p_inds, n_inds

def GAP_WithotMaxes(feat2d, p=0.05):
    b, c, w, h = feat2d.shape
    firstIndex = int(w * h * p)

    feat1d = feat2d.view(b, c, -1)
    norms = torch.norm(feat1d, p=2, dim=1)
    _, ind = torch.sort(norms, dim=1)
    s_feat1d = torch.stack([feat1d[i, :, ind[i]] for i in range(b)])
    return s_feat1d[:, :, :-firstIndex].mean(dim=2)
