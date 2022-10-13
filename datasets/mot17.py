import configparser
import errno
import glob
import itertools
import os
import typing

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TRAIN_TRANSFORMS = A.Compose(
    [
        A.SmallestMaxSize(max_size=720, interpolation=cv2.INTER_CUBIC, p=1.0),
        A.RandomCrop(height=512, width=512, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ToGray(p=0.2)
    ],
    bbox_params=A.BboxParams(format='coco')
)


class MOT17(Dataset):
    """
    A PyTorch Dataset object for reading the MOT17 dataset.
    """

    def __init__(self,
                 root_folder: str,
                 subset: str,
                 numframes: int = 2,
                 ):
        super(MOT17, self).__init__()
        self.root_folder = root_folder
        self.numframes = numframes
        self.subset = subset
        self._seqs = self._get_seqs()
        self._image_chunks, self._gt_groups, self._gt_indices, self._seq_info \
            = \
            self._get_image_chunks()
        self._transform = MOT17Transforms().get_transform(self.subset)

    def __len__(self):
        return len(self.image_chunks)

    def __getitem__(self, item):
        image_batch = self.image_chunks[item]
        gt_grp = self.gt_groups[self.gt_indices[item]]
        frame_nums = [
            self.frame_number_from_filename(x) for x in image_batch
        ]
        images = list(
            map(
                lambda x: cv2.imread(x),
                image_batch
            )
        )

        height = images[0].shape[0]
        width = images[0].shape[1]

        ground_truth = list(
            map(
                lambda x: self._gt_as_numpy(gt_grp, x, height, width),
                frame_nums
            )
        )

        image_transformed = list()
        gt_transformed = list()
        paddings = list()
        i = 0
        saved_transform = None
        for img, gt in zip(images, ground_truth):
            if i == 0:
                repeat = True
                while repeat:
                    transformed_data = self._transform(image=img, bboxes=gt)
                    if np.array(transformed_data['bboxes']).shape[0] > 0:
                        repeat = False

                saved_transform = transformed_data['replay']

            else:
                repeat = True
                while repeat:
                    transformed_data = A.ReplayCompose.replay(saved_transform, image=img, bboxes=gt)
                    if np.array(transformed_data['bboxes']).shape[0] > 0:
                        repeat = False

            image_transformed.append(transformed_data['image'])
            boxes = transformed_data['bboxes']
            gt_transformed.append(np.array(boxes))
            i += 1

        image_transformed = np.stack(
            image_transformed,
            axis=0
        )
        image_transformed = np.transpose(
            image_transformed,
            axes=[0, 3, 1, 2]
        )
        image_transformed = torch.from_numpy(image_transformed)
        for index, gt in enumerate(gt_transformed):
            num_gt = gt.shape[0]
            to_pad = 200 - num_gt
            gt_transformed[index] = np.concatenate(
                [
                    gt,
                    np.zeros((to_pad, 5), dtype=np.float32)
                ]
            )
            paddings.append(to_pad)

        gt_transformed = np.stack(
            gt_transformed,
            axis=0
        )
        gt_transformed = torch.from_numpy(gt_transformed)
        paddings = torch.from_numpy(
            np.array(paddings)
        )
        return image_transformed, gt_transformed, paddings

    @staticmethod
    def _gt_as_numpy(df_group, framenum, frame_height, frame_width):

        df = df_group.get_group(framenum)

        gt = np.stack(
            [
                df['xmin'] / frame_width,
                df['ymin'] / frame_height,
                (df['xmin'] + df['width']) / frame_width,
                (df['ymin'] + df['height']) / frame_height,
                df['tracking_id']
            ],
            axis=1
        )
        return gt

    @property
    def root_folder(self):
        return self._root_folder

    @root_folder.setter
    def root_folder(self, value):
        if not os.path.isdir(value):
            raise ValueError('The root folder path {} was not found.'.format(
                value))

        self._root_folder = value

    @property
    def subset(self):
        return self._subset

    @subset.setter
    def subset(self, value):
        if value not in ['train', 'val', 'test']:
            raise ValueError(
                'subset must be one of "train", "val" and "test".')

        self._subset = value

    @property
    def seq_file(self):
        return os.path.join(
            self.root_folder,
            'MOT17-{}.txt'.format(self.subset)
        )

    @property
    def seqs(self):
        return self._seqs

    def _get_seqs(self):
        seqs = list()
        for line in open(self.seq_file, 'r'):
            line = line.strip()
            seqs.append(line)

        return seqs

    @property
    def numframes(self):
        return self._numframes

    @numframes.setter
    def numframes(self, value):
        if not value > 1:
            raise ValueError('numframes must be a positive integer greater '
                             'than 1.')
        self._numframes = value

    @staticmethod
    def get_mot_seq_info(seq_folder: str) -> typing.Dict:
        """
        Given teh full path to a MOT17 video sequence folder, reads basic
        information
        about the video from a seqinfo.ini file located in the folder.
        :param seq_folder: (str) Full path to a MOT17 video sequence folder.
        :return: A dictionary containing metadata about the video sequence.
        """
        config = configparser.ConfigParser()
        ini_file_name = os.path.join(seq_folder, 'seqinfo.ini')
        config.read(ini_file_name)
        seq_info = dict()
        seq_info['frameRate'] = config['Sequence']['frameRate']
        seq_info['seqLength'] = config['Sequence']['seqLength']
        seq_info['imWidth'] = config['Sequence']['imWidth']
        seq_info['imHeight'] = config['Sequence']['imHeight']
        seq_info['imExt'] = config['Sequence']['imExt']
        return seq_info

    def group_images_in_chunks(self, image_list: typing.List[str]):
        """
        Given a list of images, it groups them in successive chunks.
        For example, given a list ['a','b','c','d','e'] and self.numframes=2, its
        output will be
        [['a','b'],['c','d'],['e']]. This function does suffer from the runt
        problem i.e the last chunk might not contain self.numframes elements.
        However this is not a problem in our case as we ignore such a list.
        :param image_list: A list of image files.
        :return: A list of lists such that each sublist contains self.numframes
        elements.
        """
        c = itertools.count()
        grouped_list = [list(it) for _, it in
                        itertools.groupby(image_list,
                                          lambda x: next(c) // self.numframes)]
        for g in grouped_list:
            if len(g) != self.numframes:
                grouped_list.remove(g)
        return grouped_list

    def get_images(self, seq_path: str) -> typing.List[str]:
        """
        Given the full path to a video sequence in MOT17, returns a list of
        all frame image files for the sequence. The list is sorted by frame
        numbers in ascending order.
        :param seq_path: (str) Full path to a video sequence in MOT17
        :return: A list of all frame image files for seq_path sorted by
        frame numbers in ascending order.
        """
        seq_folder = os.path.join(self.root_folder, 'MOT17-{}'.format(
            self.subset), seq_path)
        img_folder = os.path.join(
            seq_folder,
            'img1'
        )
        img_extension = self.get_mot_seq_info(seq_folder)['imExt']

        images = glob.glob(
            os.path.join(
                img_folder,
                '*{}'.format(img_extension)
            )
        )

        images.sort(
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        return images

    def _get_image_chunks(self):
        """
        Creates a list of lists such that each sublist is of length
        self.numframes. Each sublist represents a batch of frames.
        During data reading, a sublist is picked up and then its
        GT is read and returned.
        :return: A list of lists such that each sublist is of length
        self.numframes.
        """
        image_chunks = list()
        gt_groups = list()
        gt_index = list()
        seq_info = list()
        for i, seq in enumerate(self.seqs):
            seq = os.path.join(
                self.root_folder,
                'MOT17-{}'.format(self.subset),
                seq
            )
            images = self.get_images(seq)
            chunks = self.group_images_in_chunks(images)
            image_chunks.extend(chunks)
            gt_grp = self.read_gt(seq)
            gt_groups.append(gt_grp)
            seq_info.append(self.get_mot_seq_info(seq))
            gt_index.extend([i] * len(chunks))
        return image_chunks, gt_groups, gt_index, seq_info

    @property
    def image_chunks(self):
        return self._image_chunks

    @staticmethod
    def frame_number_from_filename(filename: str):
        """
        Returns the frame number from the filename of an image frame file.
        :param filename: (str) The full path to a frame image.
        :return: (int) The corresponding frame number
        """
        return int(os.path.splitext(os.path.basename(filename))[0])

    @staticmethod
    def read_gt(seq_folder: str):
        """
        Reads the GT for a specific video sequence in MOT17 dataset.
        :param seq_folder: (str) Full path to a video sequence in MOT17 dataset
        :return: A Pandas GroupBy object grouping the GT by frame numbers.
        """
        gt_file = os.path.join(
            seq_folder,
            'gt', 'gt.txt'
        )

        if not os.path.exists(gt_file):
            return None

        df = pd.read_csv(
            gt_file,
            sep=',',
            names=['frame_id', 'tracking_id', 'xmin', 'ymin', 'width', 'height',
                   'ignore_zero', 'class_id', 'confidence'],
            engine='c'
        )
        df = df[df.ignore_zero != 0]
        df = df[df.tracking_id >= 0]
        df = df.drop(df.index[~df['class_id'].isin([1, 2, 7])])
        df = df.drop(df.index[df['confidence'] < 0.5])
        df_grp = df.groupby(['frame_id'], sort=True)
        return df_grp

    @property
    def gt_groups(self):
        return self._gt_groups

    @property
    def gt_indices(self):
        return self._gt_indices

    @property
    def seq_info(self):
        return self._seq_info


class MOT17Transforms(object):
    """
    A helper class which organizes the albumentations transforms used for
    data augmentation for the MOT17 dataset.
    The relevant transform can be obtained from the get_transform(.) method.
    """

    def _trainval_transform(self):
        """
        Returns the albumentations.ReplayCompose transform for training or
        validation subsets
        :return: An albumentations.ReplayCompose object
        """

        transforms = A.ReplayCompose(
            [
                A.SmallestMaxSize(max_size=720, interpolation=cv2.INTER_CUBIC, p=1.0),
                A.RandomCrop(height=512, width=512, p=1.0),
                A.ToGray(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomScale(p=0.3),
                A.HorizontalFlip(p=0.5),
                A.Normalize(p=1.0)
            ],
            bbox_params=A.BboxParams(format='albumentations', min_visibility=0.2)
        )
        return transforms

    def _test_transform(self):
        """
        Returns the albumentations.ReplayCompose transform for the testing
        subset
        :return: An albumentations.ReplayCompose object
        """
        transforms = A.ReplayCompose(
            [
                A.Resize(height=512, width=512, interpolation=cv2.INTER_CUBIC,
                         always_apply=True),
                A.Normalize(p=1.0)
            ],
            bbox_params=A.BboxParams(format='albumentations', min_visibility=0.0)
        )
        return transforms

    def get_transform(self, subset):
        """
        Returns an albumentations.ReplayCompose object representing the
        appropriate transforms for a "subset" of the MOT17 dataset.
        The same transform object is returned for the training and the
        validation subsets but a simpler different transform is returned for
        the testing subset.
        :param subset: (str) One of "train", "val" and "test"
        :return: An albumentations.ReplayCompose object
        """
        if subset in ['train', 'val']:
            return self._trainval_transform()
        else:
            return self._test_transform()


def prepare_mot17_trackeval(mot17_folder):
    """
    Given the root folder containing the MOT17 dataset after unpacking, this
    function performs the following operations.

    a) Rename the train subfolder as MOT17-train.
    b) Rename the test subfolder as MOT17-test.
    c) Create the validation subfolder MOT17-val.
    d) Create a symbolic link to the sequence 'MOT17-01-SDP' inside the
    validation subfolder MOT17-val.
    e) Create a file MOT17-train.txt inside the root folder. This file contains
    the names of sequences to be used for training. We use the SDP suffixed
    videos for training.
    e) Create a file MOT17-val.txt inside the root folder. This file contains
    the sequence used for validation.
    f) Create a file MOT17-test.txt inside the root folder. This file contains
    the sequences used for testing.

    If the above operations have been carried out before, this function
    will not repeat the process.

    This reconditioning of the dataset folder and file structure is in
    accordance with the latest version of the MOT devkit (as of Oct 27,2021)
    which assumes the above folder structure.

    :param mot17_folder: (str) The root folder containing the MOT17 dataset.
    :return: None
    """
    if not os.path.isdir(mot17_folder):
        raise NotADirectoryError('The folder {} was not found.'.format(
            mot17_folder))

    if os.path.isdir(
            os.path.join(
                mot17_folder,
                'train'
            )
    ):
        os.rename(
            os.path.join(
                mot17_folder,
                'train'
            ),
            os.path.join(
                mot17_folder,
                'MOT17-train'
            )
        )

    if os.path.isdir(
            os.path.join(
                mot17_folder,
                'test'
            )
    ):
        os.rename(
            os.path.join(
                mot17_folder,
                'test'
            ),
            os.path.join(
                mot17_folder,
                'MOT17-test'
            )
        )

    os.makedirs(
        os.path.join(
            mot17_folder,
            'MOT17-val'
        ),
        exist_ok=True
    )

    try:
        os.symlink(
            os.path.join(
                mot17_folder,
                'MOT17-train',
                'MOT17-02-SDP'
            ),
            os.path.join(
                mot17_folder,
                'MOT17-val',
                'MOT17-02-SDP'
            ),
            target_is_directory=True
        )
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass

    train_seqs = get_train_seq(mot17_folder, root_only=True)

    val_seqs = ['MOT17-02-SDP']

    train_seq_filename = os.path.join(
        mot17_folder, 'MOT17-train.txt'
    )
    with open(train_seq_filename, 'w') as fid:
        for seq in train_seqs:
            fid.write('{}\n'.format(seq))

    val_seq_filename = os.path.join(
        mot17_folder, 'MOT17-val.txt'
    )
    with open(val_seq_filename, 'w') as fid:
        for seq in val_seqs:
            fid.write('{}\n'.format(seq))

    test_seq_filename = os.path.join(
        mot17_folder, 'MOT17-test.txt'
    )

    test_seq = get_test_seq(mot17_folder, root_only=True)

    with open(test_seq_filename, 'w') as fid:
        for seq in test_seq:
            fid.write('{}\n'.format(seq))

    return None


def get_train_seq(mot17_folder, root_only=False):
    train_folder = os.path.join(mot17_folder, 'MOT17-train')
    train_seqs = glob.glob(os.path.join(train_folder, '*SDP'))
    if root_only:
        train_seqs = list(
            map(
                lambda x: os.path.basename(x),
                train_seqs
            )
        )

    if root_only:
        train_seqs = list(set(train_seqs).difference(set(('MOT17-02-SDP'))))
    else:
        train_seqs = list(
            set(
                train_seqs
            ).difference(
                set(
                    (
                        os.path.join(mot17_folder, 'MOT17-train',
                                     'MOT17-02-SDP')
                    )
                )
            )
        )
    return train_seqs


def get_test_seq(mot17_folder, root_only=False):
    test_folder = os.path.join(mot17_folder, 'MOT17-test')
    test_seqs = glob.glob(os.path.join(test_folder, '*SDP'))
    if root_only:
        test_seqs = list(
            map(
                lambda x: os.path.basename(x),
                test_seqs
            )
        )
    return test_seqs