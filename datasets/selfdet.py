# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
pre-training dataset which implements random query patch detection.
"""
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np
import sys
sys.path.append('/root/fssd/test/DETReg')
import datasets.transforms as T

from torchvision.transforms import transforms
from PIL import ImageFilter
import random
import cv2
from util.box_ops import crop_bbox


def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
    x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
    patch = img.crop((x, y, x + sw, y + sh))
    return patch, x, y, sw, sh


class SelfDet(Dataset):
    """
    SelfDet is a dataset class which implements random query patch detection.
    It randomly crops patches as queries from the given image with the corresponding bounding box.
    The format of the bounding box is same to COCO.
    """

    def __init__(self, root, detection_transform, query_transform, cache_dir='/fssd/miniImageNet_ss_npy', max_prop=30, strategy='topk'):
        super(SelfDet, self).__init__()
        self.strategy = strategy
        self.cache_dir = cache_dir
        self.query_transform = query_transform
        self.root = root
        self.max_prop = max_prop
        self.detection_transform = detection_transform
        self.files = []
        self.dist2 = -np.log(np.arange(1, 301) / 301) / 10
        max_prob = (-np.log(1 / 1001)) ** 4

        for (troot, _, files) in os.walk(root, followlinks=True):
            for f in files:
                if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                    path = os.path.join(troot, f)
                    self.files.append(path)
                else:
                    continue
        print(f'num of files:{len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img_path = self.files[item]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 从缓存加载或计算并保存
        boxes = self.load_from_cache(item, img, h, w)

        if len(boxes) < 2:
            return self.__getitem__(random.randint(0, len(self.files) - 1))

        patches = [img.crop([b[0], b[1], b[2], b[3]]) for b in boxes]
        target = {
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)])
        }
        target['patches'] = torch.stack([self.query_transform(p) for p in patches], dim=0)
        target['boxes'] = torch.tensor(boxes)
        target['iscrowd'] = torch.zeros(len(target['boxes']))
        target['area'] = target['boxes'][..., 2] * target['boxes'][..., 3]
        target['labels'] = torch.ones(len(target['boxes'])).long()

        img, target = self.detection_transform(img, target)

        if len(target['boxes']) < 2:
            return self.__getitem__(random.randint(0, len(self.files) - 1))

        return img, target

    def load_from_cache(self, item, img, h, w):
        # 构建缓存文件名
        fn = self.files[item].split('/')[-1].split('.')[0] + '.npy'
        fp = os.path.join(self.cache_dir, fn)

        try:
            # 如果缓存存在，直接加载
            with open(fp, 'rb') as f:
                boxes = np.load(f)
            boxes=boxes[:self.max_prop]
        except FileNotFoundError:
            # 否则运行 selective search 并保存
            boxes = selective_search(img, h, w, res_size=128)
            # 只保留前 max_prop 个提议
            with open(fp, 'wb') as f:
                np.save(f, boxes)
            boxes = boxes[:self.max_prop]
        return boxes

def selective_search(img, h, w, res_size=128):
    img_det = np.array(img)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if res_size is not None:
        img_det = cv2.resize(img_det, (res_size, res_size))

    ss.setBaseImage(img_det)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process().astype('float32')

    if res_size is not None:
        boxes /= res_size
        boxes *= np.array([w, h, w, h])

    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


def make_self_det_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # The image of ImageNet is relatively small.
    scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=600),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=600),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([480], max_size=600),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_query_transforms(image_set):
    if image_set == 'train':
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    if image_set == 'val':
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    raise ValueError(f'unknown {image_set}')


def build_selfdet(image_set, args, p):
    return SelfDet(p, detection_transform=make_self_det_transforms(image_set), query_transform=get_query_transforms(image_set), cache_dir=args.cache_path,
                   max_prop=args.max_prop, strategy=args.strategy)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Generate cache for SelfDet dataset')
    parser.add_argument('--root', type=str, required=True, help='Path to image folder')
    parser.add_argument('--cache_dir', type=str, required=True, help='Directory to save/cache boxes')
    parser.add_argument('--max_prop', type=int, default=30, help='Max number of proposals per image (topk)')
    parser.add_argument('--strategy', type=str, default='topk', choices=['topk', 'mc', 'random'],
                        help='Strategy for selecting patches')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of processes for caching')

    args = parser.parse_args()

    # 创建 SelfDet 数据集实例（只用于获取文件列表）
    dummy_dataset = SelfDet(
        root=args.root,
        detection_transform=None,  # 不需要 transform
        query_transform=None,
        cache_dir=args.cache_dir,
        max_prop=args.max_prop,
        strategy=args.strategy
    )

    print(f"Start caching boxes for {len(dummy_dataset)} images using {args.workers} processes...")

    def process_index(i):
        img_path = dummy_dataset.files[i]
        fn = os.path.basename(img_path).split('.')[0] + '.npy'
        fp = os.path.join(args.cache_dir, fn)

        if os.path.exists(fp):  # 跳过已有缓存
            return

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            boxes = selective_search(img, h, w, res_size=128)
            boxes = boxes[:args.max_prop]

            with open(fp, 'wb') as f:
                np.save(f, boxes)
        except Exception as e:
            print(f"Error processing index {i}: {e}")

    # 使用 ProcessPoolExecutor 多进程处理，并结合 tqdm 展示进度
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(process_index, range(len(dummy_dataset))), total=len(dummy_dataset)))

    print("Caching completed.")