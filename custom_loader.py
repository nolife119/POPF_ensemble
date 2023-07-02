import enum
import os
import random
import numbers

import pandas as pd
import numpy as np
from PIL import Image
import torch.utils.data
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import cv2

from common import *
from preprocessor import crop_2d


class DataPurpose(enum.Enum):
    Train = 'Train'
    Validation = 'Validation'
    Test = 'Test'
    # Dev = 'Development'
    Total = 'Total'

class ModelTarget(enum.Enum):
    POPF = 'popf'
    CR_POPF = 'cr-popf'


class ImageDataset(Dataset):
    def __init__(self, data_purpose, config=None):
        if config is None:
            config = ConfigManager().load()
        self.img_max = int(config["img_max"])
        self.trans_prob = float(config["trans_prob"])
        self.config = config
        self.data_purpose = data_purpose
        model_target = config["model_target"].lower()
        self.model_target = model_target

        case_info_path = config['case_info_path']
        case_info = pd.read_csv(case_info_path)

        tv_split_m = int(config["tv_split_m"])

        pid2path, pid2trg = dict(), dict()
        for idx,row in case_info.iterrows():
            pid, op_date, is_popf, is_crpopf = row[:4]
            y, m, d = map(int, op_date.split('-'))
            if data_purpose == DataPurpose.Train and (y == 2017 or y == 2016) and m <= tv_split_m:
                pid2path[pid] = []
            elif data_purpose == DataPurpose.Validation and (y == 2017 or y == 2016) and tv_split_m < m:
                pid2path[pid] = []
            elif data_purpose == DataPurpose.Test and y == 2018:
                pid2path[pid] = []
            elif data_purpose == DataPurpose.Total:
                pid2path[pid] = []
            pid2trg[pid] = [is_popf, is_crpopf]

        # get centroid info
        centroid_info = pd.read_csv(config['centroid_path'])
        slice_gap = int(config['slice_gap'])
        pid2centroid = dict()
        for idx, row in centroid_info.iterrows():
            pid = row[0]
            pid2centroid[pid] = list(map(int, row[1:4]))

        # make pid 2 path dictionary
        img_dir = config['jpg_root']
        temp_dic = dict()
        path_lst, target_lst = [], []
        for f_name in os.listdir(img_dir):
            if not f_name.endswith('.jpg'):
                continue
            name_only = f_name.split('.')[0]
            now_id, img_idx = map(int, name_only.split('_'))
            if now_id in pid2path:
                path = '%s/%s' % (img_dir, f_name)
                if now_id not in temp_dic:
                    temp_dic[now_id] = []
                temp_dic[now_id].append([img_idx, path])

        for pid, args_lst in temp_dic.items():
            args_lst = sorted(args_lst)
            p_lst = np.array(args_lst)[:, 1]

            is_popf, is_crpopf = pid2trg[pid]
            z = pid2centroid[pid][0]

            for img_idx in range(len(p_lst)):
                if z - slice_gap <= img_idx + 1 <= z + slice_gap:
                    path = p_lst[img_idx]
                    pid2path[pid].append(path)
                    target_lst.append([is_popf, is_crpopf])
                    path_lst.append(path)

        self.pid2path = pid2path
        self.path_lst = path_lst
        self.target_lst = target_lst
        alpha, beta = 10, 0.05
        self.transform = RandomAffine(degrees=alpha, translate=[beta, beta],
                                      scale=[1 - beta, 1 + beta], shear=[-1 - beta, 1 + beta, -1 - beta, 1 + beta])

        self.use_cropping = True if config['use_cropping'].lower() == 'true' else False
        self.pid2centroid = pid2centroid
        self.img_shape = list(map(int, config['img_shape'].split(',')))

    def __len__(self):
        return len(self.path_lst)

    def __getitem__(self, index):
        img_path = self.path_lst[index]
        # target = self.target_lst[index]
        image = Image.open(img_path)
        if self.data_purpose == DataPurpose.Train and random.random() <= self.trans_prob:
            image = self.transform(image)

        image = np.array(image)[:, :, 0]
        image = image / self.img_max
        image[image < 0] = 0.
        image[image > 1] = 1.

        if self.use_cropping:
            pid, _ = parse_path(img_path)
            centroid = self.pid2centroid[pid][1:]
            # centroid = [320, 448]
            image = crop_2d(image, centroid, self.img_shape)

        image = np.expand_dims(image, 0)
        t1, t2 = self.target_lst[index]
        t1 = np.array(t1).astype(np.int32)
        t2 = np.array(t2).astype(np.int32)
        return image, t1, t2


def parse_path(file_path):
    f_name = os.path.basename(file_path)
    n_only = f_name.split('.')[0]
    pid, idx = list(map(int, n_only.split('_')))
    return pid, idx


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img2 = F.affine(img, *ret)
        return img2



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        label_lst = []
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
            label_lst.append(label)

        # weight for each sample
        weights = [1.0 / label_to_count[label_lst[idx]]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.target_lst[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def load_pkl_and_show():
    input_dir = 'D:/Data/POPF_data/POPF_pp3d'
    name_lst = os.listdir(input_dir)
    indexes = list(range(len(name_lst)))
    random.shuffle(indexes)
    idx = indexes[0]
    name = '4062.pkl'
    pid = int(name.split('.')[0])

    path = '%s/%s' % (input_dir, name)
    data_dict = load_obj(path)
    print(data_dict)
    img, roi = data_dict['volume'], data_dict['roi']
    msg = 'now pid: %s' % pid
    StackViewer(img, roi, msg).show()

