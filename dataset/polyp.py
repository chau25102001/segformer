import os
import torchvision
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.augmentations import transforms, crops
from albumentations.augmentations.geometric import rotate, resize
from albumentations.core.composition import Compose, OneOf
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as tf


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image).float()
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Compose_Transform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PolypDataset(Dataset):
    """
    img_root_path/
        img_subpath/
            img0.png
            .....
     el_subpath/
            label0.png
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(
            self,
            root_path,
            img_subpath,
            label_subpath,
            img_size,
            use_aug,
            use_cutmix=0.5,
            cache_train=True,
    ):
        assert cache_train, "Please use cache train"
        if isinstance(root_path, str):
            root_path = [root_path]
        items = []
        for root in root_path:
            image_path = os.path.join(root, img_subpath)
            label_path = os.path.join(root, label_subpath)
            # print(image_path)
            imp = set(os.listdir(image_path))
            lap = set(os.listdir(label_path))
            sample = list(set.intersection(imp, lap))
            items.extend([(os.path.join(image_path, p), os.path.join(label_path, p)) for p in sample])
        self.img_size = img_size
        self.cache_train = cache_train
        self.use_aug = use_aug
        self.use_cutmix = use_cutmix
        self.items = None
        if cache_train:
            temp = []
            path = []
            for ip, lp in tqdm(items, desc="Caching data"):
                i = self.load_img(ip, img_size)
                l = self.load_img(lp, img_size)
                if i is not None and l is not None:
                    temp.append((i, l))
                    path.append((ip, lp))

            self.items = temp
            self.items_path = path

    def __len__(self):
        return len(self.items_path)

    def __getitem__(self, item):
        if self.cache_train:
            img, label = self.items[item]
        else:
            img = self.load_img(self.items_path[item][0], self.img_size)
            label = self.load_img(self.items_path[item][1], self.img_size)
        if self.use_cutmix > 0 and random.uniform(0, 1) < self.use_cutmix:
            mix_index = random.randint(0, len(self) - 1)
            if self.cache_train:
                img2, label2 = self.items[mix_index]
            else:
                img2 = self.load_img(self.items_path[mix_index][0], self.img_size)
                label2 = self.load_img(self.items_path[mix_index][1], self.img_size)
            img, label = self.cutmix(img, img2, label, label2)
        label = np.array(label, dtype=np.uint8)
        img = img.astype(np.uint8)
        img, label = self.transform(img, label, self.img_size, self.img_size, self.use_aug)
        return img, label

    def load_img(self, path, size, gray=False):
        # Only load square img
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not gray else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
            return img
        except Exception:
            print(f"WARNING: Cant load image from {path}")
            return None

    @staticmethod
    def normalize(img, l=None):
        img -= PolypDataset.mean
        img /= PolypDataset.std
        return img

    @staticmethod
    def denormlize(img):
        img *= PolypDataset.std
        img += PolypDataset.mean
        return img

    @staticmethod
    def transform(img, label, h, w, use_aug):
        train_transform = Compose([
            rotate.RandomRotate90(),
            transforms.Flip(),
            transforms.HueSaturationValue(),
            transforms.RandomBrightnessContrast(),
            transforms.Transpose(),
            OneOf([
                crops.RandomCrop(224, 224, p=1),
                crops.CenterCrop(224, 224, p=1)
            ], p=0.2),
            resize.Resize(h, w)
        ], p=0.5)
        img_transform = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(PolypDataset.mean, PolypDataset.std)
        ])
        label_transform = tf.Compose([
            tf.ToTensor()
        ])
        if use_aug:
            out = train_transform(image=img, mask=label)
            img, label = out["image"], out["mask"]
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = np.where(label > 127, 1, 0)
        img = img_transform(img)
        label = label_transform(label)
        return img, label.float()

    @staticmethod
    def cutmix(img0, img1, target0, target1):
        size0 = img0.shape
        size1 = img1.shape
        min_size = min(size0[0], size1[0]), min(size0[1], size1[1]), 3
        mask_size = min_size[0] // 2 - 1, min_size[1] // 2 - 1  # w, h
        point_mask = random.randint(0, min_size[0] - mask_size[0]), random.randint(0, min_size[1] - mask_size[1])
        mask_arr = np.ones(min_size)
        mask_arr[point_mask[0]:point_mask[0] + mask_size[0], point_mask[1]:point_mask[1] + mask_size[1], :] = 0
        mix_img = mask_arr * img0[:min_size[0], : min_size[1], :] + (1 - mask_arr) * img1[:min_size[0], : min_size[1],
                                                                                     :]
        mix_target = mask_arr * target0[:min_size[0], : min_size[1], :] + (1 - mask_arr) * target1[:min_size[0],
                                                                                           : min_size[1], :]
        return mix_img, mix_target

    @staticmethod
    def pad(img, label, new_size):
        """
        img: numpy
        new_size: W, H
        """
        img = Image.fromarray(img)
        label = Image.fromarray(label)
        old_size = img.size  # W, H
        assert old_size == label.size
        w_pad = new_size[0] - old_size[0]
        h_pad = new_size[1] - old_size[1]
        if w_pad <= 0 or h_pad <= 0:
            return img.resize((new_size[0], new_size[1])), label.resize((new_size[0], new_size[1]))
        w_pad_0 = random.randint(0, w_pad)
        w_pad_1 = int(w_pad - w_pad_0)
        h_pad_0 = random.randint(0, h_pad)
        h_pad_1 = int(h_pad - h_pad_0)
        tf = torchvision.transforms.Pad((h_pad_0, w_pad_0, h_pad_1, w_pad_1))
        return tf(img), tf(label)


#
# class Multiscale_collate_fn:
#     default_scale = [0.75, 1, 1.25]
#
#     def __init__(self, scales=None):
#         self.scales = scales if scales is not None else Multiscale_collate_fn.default_scale
#
#     @staticmethod
#     def transform(img, label):
#         tf = Compose_Transform([
#             PILToTensor(),
#             Normalize(PolybDatset.mean, PolybDatset.std)
#         ])
#         return tf(img, label)
#
#     def __call__(self, batch):
#         imgs, labels = zip(*batch)
#         hw = []
#         for im in imgs:
#             scale = random.choice(self.scales)
#             hw.append([im.shape[0] * scale, im.shape[1] * scale])
#         max_w = max(hw, key=lambda x: x[1])[1]
#         max_h = max(hw, key=lambda x: x[0])[0]
#         data = [PolybDatset.pad(img, label, (max_w, max_h)) for img, label in zip(imgs, labels)]
#         imgs, labels = zip(*[self.transform(im, la) for im, la in data])
#         return torch.stack(imgs), torch.stack(labels)
#

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # img = Image.open("/home/crazylov3/LTGiang/MDEQ-Vision/TestDataset/CVC-300/images/149.png")
    # plt.imshow(img)
    # plt.show()
    # exit()
    dataset = PolypDataset(
        root_path="/home/crazylov3/LTGiang/MDEQ-Vision/TestDataset/CVC-300",
        img_subpath="images",
        label_subpath="masks",
        img_size=352,
        use_aug=True,
        use_cutmix=False,
        # cache_train=True
    )
    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        # collate_fn=Multiscale_collate_fn()
    )
    samples, labels = next(iter(loader))
    for _img, _label in zip(samples, labels):
        fix, ax = plt.subplots(1, 2, figsize=(10, 15))
        # _img, _label = dataset[random.randint(0, len(dataset) - 1)]
        # _img = PolybDatset.denormlize(_img.permute(1, 2, 0)).numpy()
        print(_img.dtype)
        print(_label.dtype)
        exit()
        _img = _img.permute(1, 2, 0).numpy()
        _img = PolypDataset.denormlize(_img)
        # print(_img.shape)
        ax[0].imshow(_img)
        ax[1].imshow(_label[0])
        plt.show()
        # print(type(_label))
        # exit()
