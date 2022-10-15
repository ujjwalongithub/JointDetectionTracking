import argparse

from torch.utils.data import DataLoader

import colors
from datasets.mot17 import MOT17

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
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def denormalize_image(image):
    return UnNormalize()(image)


def get_unique_tracks(gt):
    unique_ids = set(gt[0, :, :, -1].flatten())
    unique_ids = unique_ids.difference([0])
    return list(unique_ids)


def prepare_color_dict(gt):
    unique_ids = get_unique_tracks(gt)
    unique_colors = colors.distinct_colors(len(unique_ids))
    color_dict = dict(zip(unique_ids, unique_colors))
    return color_dict


def denormalize_boxes(gt, padding):
    boxes = gt[0, :, :(201 - padding), :]
    boxes[:, :, :4] *= 512
    return boxes


if __name__ == "__main__":
    args = parser.parse_args()
    db = MOT17(
        root_folder=args.root_path,
        subset=args.subset,
        numframes=args.num_frames
    )

    loader = DataLoader(dataset=db, batch_size=1,
                        shuffle=True)

    for data in loader:
        image, bbox, paddings = data
        print(bbox.shape)
        # print(image)
        # print('-----')
        # image2 = denormalize_image(image)
        # image_to_write = image2[0,0,:,:,:]
        # image_to_write = np.transpose(image_to_write,(1,2,0))
        # print(image_to_write.shape)
        # image_to_write = image_to_write.numpy()
        # normalized_image = np.ascontiguousarray(np.zeros_like(image_to_write))
        # print(normalized_image.shape)
        # normalized_image = cv2.normalize(image_to_write, normalized_image,0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('abc.jpg', normalized_image)
        break
