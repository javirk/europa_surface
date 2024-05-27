import os
import torch
import json
import h5py
import numpy as np
from glob import glob
from pathlib import Path
from torchvision import tv_tensors
from torchvision.transforms import v2
from segment_anything.utils.transforms import ResizeLongestSide

from src.datasets.base import DatasetBase, plot_return
from src.datasets.transforms import GaussianNoise
from src.datasets.dataset_utils import none_collate


class GalileoDataset(DatasetBase):
    slices_domain = {
        "background": 0,
        "band": 1,
        "double_ridge": 2,
        "ridge_complex": 3,
        "undifferentiated_linea": 4
    }
    num_classes = len(slices_domain.values())
    ignore_index = 5
    image_size = 224

    def __init__(self, root, split, dataset_type='all', transforms=None, bbox_shift=20, instance_segmentation=False,
                 **kwargs):
        assert split in ['train', 'val', 'test', 'trainval']
        assert dataset_type in ['new', 'old', 'all']
        # self.root = root
        self.transforms = transforms
        self.split = split
        self.imgs = self.get_imgs_names(root, split, dataset_type)

        self.sam_transform = ResizeLongestSide(1024)  # Images are 1024x1024 in SAM
        self.bbox_shift = bbox_shift
        self.downsampling_size = self.image_size // 4
        self.instance_segmentation = instance_segmentation

    @staticmethod
    def legacy_get_imgs_names(root, split):
        path = os.path.join(root, split)
        data = sorted(glob(os.path.join(path, '*.hdf5')))
        assert len(data) > 0, f'No data found'
        return data

    @staticmethod
    def get_imgs_names(root, split, dataset_type):
        file = os.path.join(root, 'splits.json')
        # Load the json file
        with open(file, 'r') as f:
            data = json.load(f)
        dataset_data = data[dataset_type][split]
        dataset_data = [os.path.join(root, x) for x in dataset_data]
        return dataset_data

    def _read_image(self, img_data):
        """
        Reads the H5 file and returns a 3 dimensional or grayscale Tensor. Equivalent to io.read_image.
        The values of the output tensor are uint8 in [0, 255].
        :param img_data:
        :return:
        """
        with h5py.File(img_data, "r") as f:
            img = f['image'][:]
            # mask_ids = f['mask_ids'][:]  # [instances,]
            bboxes = f['bboxes'][:]  # [instances, 4]
            sem_mask = f['instance_mask'][:] if self.instance_segmentation else f['semantic_mask'][:]
            instance_mask = f['instance_mask'][:]
            # Either [H, W, instances] or [H, W]

        # Instance segmentation is one-shot. Converting to normal.
        # Careful! There is not an instance for the background, so applying argmax directly is wrong
        inverse_background_mask = np.sum(instance_mask, axis=-1)
        background_mask = np.where(inverse_background_mask == 0, 1, 0)
        instance_mask = np.concatenate([background_mask[:, :, None], instance_mask], axis=-1)
        instance_mask = np.argmax(instance_mask, axis=-1)

        # if self.instance_segmentation:
        #     # Data is one-shot. Convert to normal
        #     mask = np.argmax(mask, axis=-1)
        # else:
        #     # We have to use the mask_ids on the mask. Mask comes with instance segmentation
        #     mask = mask * mask_ids
        #     mask = np.sum(mask, axis=-1)
        assert np.max(sem_mask) < 5, f"Max value in mask is {np.max(sem_mask)}"

        sem_mask = sem_mask.astype(np.float32)
        instance_mask = instance_mask.astype(np.float32)

        # tv_tensors is weird. [None] can't be applied later, or it falls back to torch.Tensor:
        # https://pytorch.org/vision/main/auto_examples/transforms/plot_tv_tensors.html#why-is-this-happening
        img = tv_tensors.Image(img[None])
        sem_mask = tv_tensors.Mask(sem_mask[None])
        bboxes = tv_tensors.BoundingBoxes(bboxes, format='xyxy', canvas_size=img.shape[-2:], dtype=torch.int64)
        instance_mask = tv_tensors.Mask(instance_mask[None])
        return img, sem_mask, instance_mask, bboxes

    def _get_name(self, img_data):
        return Path(img_data).stem


class Galileo:
    test_subset = None

    def __init__(self,
                 location=os.path.expanduser('~/data'),
                 dataset_type='all',
                 batch_size=128,
                 num_workers=8,
                 transformations=None,
                 shuffle_training=True,
                 fold_number=0,
                 **kwargs):
        args = kwargs['kwargs']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.location = location
        print("Loading Test Data from ", self.location)

        if transformations is None:
            transformations = v2.Compose([
                v2.RandomRotation(20),
                v2.RandomHorizontalFlip(),
                # GaussianNoise(mean=0, sigma=(1, 5)),
                v2.GaussianBlur(kernel_size=3),
                # v2.SanitizeBoundingBoxes(min_size=25, labels_getter=None),
            ])

        if args.training_split == 'trainval':
            training_split = 'trainval'
        else:
            training_split = 'train'

        self.train_dataset = GalileoDataset(root=location, split=training_split, dataset_type=dataset_type,
                                            fold_number=fold_number, transforms=transformations)
        self.val_dataset = GalileoDataset(root=location, split='val', dataset_type=dataset_type,
                                          fold_number=fold_number)
        self.test_dataset = GalileoDataset(root=location, split='test', dataset_type=dataset_type,
                                           fold_number=fold_number)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_training,
            num_workers=self.num_workers,
            collate_fn=none_collate
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=none_collate
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=none_collate
        )

    def __repr__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    from torchvision.transforms import v2
    from torch.utils.data import DataLoader

    trans = v2.Compose([
        v2.RandomRotation(20),
        v2.RandomHorizontalFlip(),
        # GaussianNoise(mean=0, sigma=(1, 5)),
        v2.GaussianBlur(kernel_size=3),
        # v2.SanitizeBoundingBoxes(min_size=25, labels_getter=None),
    ])

    # trans = v2.RandAugment(num_ops=3)

    root_folder = '/Users/javier/Documents/datasets/europa/'
    dataset = GalileoDataset(root_folder, 'train', transforms=trans, fold_number=0, instance_segmentation=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for i in range(len(dataset)):
        # continue
        plot_return(dataset[i])
        # break

        # Save to a file
        # plt.savefig(f'tests_mask/{i}.png')
