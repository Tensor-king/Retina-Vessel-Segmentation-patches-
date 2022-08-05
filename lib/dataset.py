import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class RandomCrop:
    def __init__(self, shape):
        self.shape = [shape, shape] if isinstance(shape, int) else shape
        self.fill = 0
        self.padding_mode = 'constant'

    @staticmethod
    def _get_range(shape, crop_shape):
        if shape == crop_shape:
            start = 0
        else:
            start = random.randint(0, shape - crop_shape)
        end = start + crop_shape
        return start, end

    def __call__(self, img, mask):
        _, h, w = img.shape
        sh, eh = self._get_range(h, self.shape[0])
        sw, ew = self._get_range(w, self.shape[1])
        return img[:, sh:eh, sw:ew], mask[:, sh:eh, sw:ew]


class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = random.uniform(0, 1)
        return self._flip(img, prob), self._flip(mask, prob)


class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob <= self.prob:
            img = img.flip(1)
        return img

    def __call__(self, img, mask):
        prob = random.uniform(0, 1)
        return self._flip(img, prob), self._flip(mask, prob)


class RandomRotate:

    def __call__(self, img, mask):
        cnt = random.randint(0, 360)
        img_tran = transforms.RandomRotation((cnt, cnt), interpolation=InterpolationMode.BILINEAR)
        mask_tran = transforms.RandomRotation((cnt, cnt), interpolation=InterpolationMode.NEAREST)
        return img_tran(img), mask_tran(mask)


class Compose:
    def __init__(self, transforms_):
        self.transforms = transforms_

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class TestDataset(Dataset):
    # 已经预处理过了
    def __init__(self, patches_imgs):
        self.imgs = patches_imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx, ...]).float()
