import argparse

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.mot17 import MOT17
from utils import colors

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='Root folder where MOT17 dataset has been stored.')
parser.add_argument('--num_frames', required=False, default=10, type=int, help='Number of frames in a batch.')
parser.add_argument('--subset', choices=['train', 'val'], default='train')


class UnNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of any size to be normalized
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_unique_tracks(gt):
    unique_ids = set(gt[0, :, :, -1].numpy().astype(int).flatten())
    unique_ids = unique_ids.difference([0])
    return list(unique_ids)


def prepare_color_dict(gt):
    unique_ids = get_unique_tracks(gt)
    unique_colors = colors.distinct_colors(len(unique_ids))
    color_dict = dict(zip(unique_ids, unique_colors))
    return color_dict


def plot_boxes(images, gt, paddings, file_name):
    imgs = list()
    images = UnNormalize()(images)
    color_dict = prepare_color_dict(gt)
    print(color_dict)
    for imgnum in range(images.shape[1]):
        img = images[0, imgnum, :, :, :].numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.ascontiguousarray(img)
        normalized_image = np.ascontiguousarray(np.zeros_like(img))
        normalized_image = cv2.normalize(img, normalized_image, 0, 255, cv2.NORM_MINMAX)
        normalized_image.astype(np.uint8)
        frame_gt = gt[0, imgnum, :, :]
        padding = paddings[0, imgnum]
        boxes = frame_gt[:(200 - padding), :]
        boxes[:, :4] *= 512
        boxes = torch.round(boxes)
        # boxes = np.round(boxes)
        boxes = boxes.numpy()
        boxes = boxes.astype(int)
        for _ in range(boxes.shape[0]):
            clr = list(color_dict[boxes[i, 4]][::-1])
            clr = [x * 255 for x in clr]
            clr = list(map(int, clr))
            img = cv2.rectangle(normalized_image, (boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), clr)

        imgs.append(img)
    imageio.mimwrite(file_name, imgs)
    return None


if __name__ == "__main__":
    args = parser.parse_args()
    db = MOT17(
        root_folder=args.root_path,
        subset=args.subset,
        numframes=args.num_frames
    )

    loader = DataLoader(dataset=db, batch_size=1,
                        shuffle=True)

    i = 1
    for data in loader:
        image, bbox, paddings = data
        filename = 'batch_{}.gif'.format(i)
        plot_boxes(image, bbox, paddings, filename)
        i += 1
