import os
import torch
import h5py
import numpy as np
from glob import glob
from torchvision import tv_tensors
from torchvision.transforms import v2
from segment_anything.utils.transforms import ResizeLongestSide

from src.datasets.base import DatasetBase
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

    def __init__(self, root, split, transforms=None, bbox_shift=20, instance_segmentation=False, **kwargs):
        assert split in ['train', 'val', 'test']
        # self.root = root
        self.transforms = transforms
        self.split = split
        self.imgs = self.get_imgs_names(root, split)

        self.sam_transform = ResizeLongestSide(1024)  # Images are 1024x1024 in SAM
        self.bbox_shift = bbox_shift
        self.downsampling_size = self.image_size // 4
        self.instance_segmentation = instance_segmentation

    @staticmethod
    def get_imgs_names(root, split):
        path = os.path.join(root, split)
        data = sorted(glob(os.path.join(path, '*.hdf5')))
        assert len(data) > 0, f'No data found'
        return data

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
            mask = f['instance_mask'][:] if self.instance_segmentation else f['semantic_mask'][:]
            # Either [H, W, instances] or [H, W]

        # if self.instance_segmentation:
        #     # Data is one-shot. Convert to normal
        #     mask = np.argmax(mask, axis=-1)
        # else:
        #     # We have to use the mask_ids on the mask. Mask comes with instance segmentation
        #     mask = mask * mask_ids
        #     mask = np.sum(mask, axis=-1)
        assert np.max(mask) < 5, f"Max value in mask is {np.max(mask)}"

        mask = mask.astype(np.float32)

        # tv_tensors is weird. [None] can't be applied later, or it falls back to torch.Tensor:
        # https://pytorch.org/vision/main/auto_examples/transforms/plot_tv_tensors.html#why-is-this-happening
        img = tv_tensors.Image(img[None])
        mask = tv_tensors.Mask(mask[None])
        bboxes = tv_tensors.BoundingBoxes(bboxes, format='xyxy', canvas_size=img.shape[-2:], dtype=torch.int64)
        return img, mask, bboxes

    def _get_name(self, img_data):
        return os.path.join(img_data[0], str(img_data[1]))


class Galileo:
    test_subset = None

    def __init__(self,
                 location=os.path.expanduser('~/data'),
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

        self.train_dataset = GalileoDataset(root=location, split='train', fold_number=fold_number,
                                            transforms=transformations)
        self.val_dataset = GalileoDataset(root=location, split='val', fold_number=fold_number)
        self.test_dataset = GalileoDataset(root=location, split='test', fold_number=fold_number)

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
    import torchvision
    import matplotlib.pyplot as plt
    from torchvision.transforms import v2
    from torch.utils.data import DataLoader


    def show_points(coords, ax, marker_size=200):
        ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.)

    trans = v2.Compose([
        v2.RandomRotation(20),
        v2.RandomHorizontalFlip(),
        # GaussianNoise(mean=0, sigma=(1, 5)),
        v2.GaussianBlur(kernel_size=3),
        # v2.SanitizeBoundingBoxes(min_size=25, labels_getter=None),
    ])

    # trans = v2.RandAugment(num_ops=3)

    root_folder = '/Users/javier/Documents/datasets/europa/dataset_224x224/'
    dataset = GalileoDataset(root_folder, 'train', transforms=trans, fold_number=0, instance_segmentation=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for i, res in enumerate(dataset):
        im = res['image']
        m_box = res['mask_bb']
        m_point = res['mask_point']
        box = res['boxes']
        point = res['point']

        # Bring back the box to the original size
        # box = box.reshape(-1, 2, 2)
        # box[..., 0] = box[..., 0] * (im.shape[-1] / 1024)
        # box[..., 1] = box[..., 1] * (im.shape[-2] / 1024)
        # box = box.reshape(-1, 4)
        # # Bring back the point to the original size
        # point[..., 0] = point[..., 0] * (im.shape[-1] / 1024)
        # point[..., 1] = point[..., 1] * (im.shape[-2] / 1024)

        # fix, ax = plt.subplots(1, 3)
        # ax[0].imshow(torchvision.utils.draw_bounding_boxes(im, box, colors='red').permute(1, 2, 0))
        # show_points(point, ax[0])
        # ax[1].imshow(m_box[0])
        # ax[2].imshow(m_point[0])
        #
        # ax[0].set_title('Original image')
        # ax[1].set_title('Mask BB')
        # ax[2].set_title('Mask Point')
        # plt.show()
        # break

        # Save to a file
        # plt.savefig(f'tests_mask/{i}.png')
