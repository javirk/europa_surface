import os
import torch
import json
import random
import numpy as np
from PIL import Image
from functools import partial
from torchvision import tv_tensors
from torchvision.transforms import v2
from segment_anything.utils.transforms import ResizeLongestSide

from src.datasets.data_info import DATASET_INFO
from src.datasets.transforms import GaussianNoise
from src.datasets.base import DatasetBase
from src.datasets.dataset_utils import none_collate

domains = ["cirrus", "spectralis", "topcon"]


class RetouchDataset(DatasetBase):
    slices_domain = {
        'cirrus': 128,
        'spectralis': 49,
        'topcon': 128
    }

    def __init__(self, root, domain, split, fold_number, transforms=None, only_fluid=False, augmenting_transforms=None,
                 bbox_shift=20, transformations_per_sample=1, add_volume_information=None, max_batch_size=64, **kwargs):
        domain = domain.lower()
        assert domain in ['cirrus', 'spectralis', 'topcon']
        assert split in ['train', 'val', 'test']
        # self.root = root
        self.transforms = transforms
        self.split = split
        img_paths = self.get_imgs_names(root, domain, split, fold_number)
        # img_paths = np.random.choice(img_paths, int(len(img_paths) * proportion))
        if only_fluid:
            self.imgs = self._read_fluid_slices(root, domain, img_paths)
        else:
            self.imgs = self._read_all_slices(root, domain, img_paths)
        self.num_classes = DATASET_INFO['retouch']['num_classes']
        self.ignore_index = DATASET_INFO['retouch']['ignore_index']

        self.sam_transform = ResizeLongestSide(1024)  # Images are 1024x1024 in SAM
        self.bbox_shift = bbox_shift
        self.transformations_per_sample = transformations_per_sample
        self.augmenting_transforms = augmenting_transforms
        self.image_size = DATASET_INFO['retouch']['training_size']
        self.downsampling_size = self.image_size // 4
        self.add_volume_information = add_volume_information
        if self.add_volume_information:
            self.num_negatives = max_batch_size - 1
            self.num_positives = 1
            self.std_dev = 2
            self.slices_idx = np.array(range(0, self.slices_domain[domain]))

    @staticmethod
    def _read_fluid_slices(root, domain, names):
        with open(os.path.join(root, f"{domain}.json"), 'r') as f:
            data = json.load(f)
        imgs = []
        for name in names:
            path = os.path.join(root, domain, name)
            list_slices = data[name]
            list_slices = [(path, x) for x in list_slices]
            imgs.extend(list_slices)
        return imgs

    @staticmethod
    def _read_all_slices(root, domain, names):
        imgs = []
        for name in names:
            path = os.path.join(root, domain, name)
            img = Image.open(os.path.join(path, 'image.tif'))
            list_slices = [(path, x) for x in range(img.n_frames)]
            imgs.extend(list_slices)
        return imgs

    @staticmethod
    def get_imgs_names(root, domain, split, fold_number):
        # Open the file
        with open(os.path.join(root, "splits", f"split_{fold_number}.json"), 'r') as f:
            # Load the JSON data
            data = json.load(f)
        data = data[domain][split]
        assert len(data) > 0, f'Split is not possible for domain {domain}'
        # data = [os.path.join(root, domain, x) for x in data]
        return data

    def _read_image(self, img_data):
        """
        Reads the TIF file and returns a 3 dimensional or grayscale Tensor. Equivalent to io.read_image.
        The values of the output tensor are uint8 in [0, 255].
        :param img_data:
        :return:
        """
        img = Image.open(os.path.join(img_data[0], 'image.tif'))
        mask = Image.open(os.path.join(img_data[0], 'label.tif'))
        img.seek(img_data[1])
        mask.seek(img_data[1])

        # tv_tensors is weird. [None] can't be applied later, or it falls back to torch.Tensor:
        # https://pytorch.org/vision/main/auto_examples/transforms/plot_tv_tensors.html#why-is-this-happening
        img = tv_tensors.Image(np.array(img)[None])
        mask = tv_tensors.Mask(np.array(mask)[None])
        return img, mask

    def _get_name(self, img_data):
        return os.path.join(img_data[0], str(img_data[1]))


class RetouchFourierDataset(RetouchDataset):
    """Return also the Fourier transform of the image"""

    def __init__(self, root, domain, split, transforms=None, only_fluid=False, bbox_shift=20,
                 transformations_per_sample=1):
        super().__init__(root, domain, split, transforms, only_fluid, bbox_shift,
                         transformations_per_sample)

    def __getitem__(self, idx):
        dict_return = super().__getitem__(idx)
        # Downsample first, then fo the FFT. The reverse will give the wrong result
        image_down = torch.nn.functional.interpolate(dict_return['image'][None], size=(128, 128), mode='bilinear')

        ffted = torch.fft.fft2(image_down[0, 0] / 255., dim=(-1, -2), norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=0)

        dict_return['fourier'] = ffted  # Return 2xHxW
        return dict_return


class RetouchEmbeddingsDataset(RetouchDataset):

    def __init__(self, root, root_base, domain='Spectralis', split='train', only_fluid=False):
        super().__init__(root_base, domain, split, proportion=1, transforms=None, only_fluid=only_fluid)
        self.embs = self._read_all_embeddings(root, domain)

    def _read_all_embeddings(self, root, domain):
        embs = []
        for path, slice_number in self.imgs:
            image_name = path.split('/')[-1]
            path = os.path.join(root, domain, image_name)
            embs.append((path, slice_number))
        return embs

    def _read_embeddings(self, emb_data):
        path = os.path.join(emb_data[0], f'{emb_data[1]}.pt')
        emb = torch.load(path, map_location='cpu')
        return emb

    def __getitem__(self, idx):
        img_data = self.imgs[idx]  # path, slice number
        emb_data = self.embs[idx]  # path, slice number
        img, mask = self._read_image(img_data)
        emb = self._read_embeddings(emb_data)

        # Downsample mask
        mask_downsampled = torch.nn.functional.interpolate(mask[None], size=(128, 128), mode='nearest')[0, 0].long()

        # Make mask binary
        mask_binary = (mask > 0).float()

        bounding_boxes, labeled_mask = self.get_bounding_boxes(mask_binary[0])
        # Select one bounding box at random. Zero all the values of the mask outside the bounding box
        bounding_box = bounding_boxes[np.random.choice(len(bounding_boxes))]
        mask_bb = self.isolate_mask_bb(bounding_box, mask_binary)
        # Transform the BB so that it can be used in SAM
        # bounding_box = self.sam_transform.apply_boxes_torch(bounding_box, mask.shape[-2:])
        if len(bounding_box.shape) == 1:
            bounding_box = bounding_box.unsqueeze(0)

        # Point stuff:
        point = np.where(mask_binary == 1)
        if len(point[0]) == 0:
            # There's no fluid. Random point and background label
            point = torch.tensor([random.randint(0, mask_binary.shape[-1]),
                                  random.randint(0, mask_binary.shape[-2])])
            mask_point = torch.zeros_like(mask_binary)
            point_label = torch.tensor([0])
        else:
            point_idx = np.random.choice(len(point[0]))
            point = torch.tensor(
                [point[2][point_idx], point[1][point_idx]])  # mask is 1xHxW, so we discard the first dim.
            mask_point = self.isolate_mask_point(point, labeled_mask)
            # Transform the point so that it can be used in SAM
            # point = self.sam_transform.apply_coords_torch(point, mask.shape[-2:]).unsqueeze(0)
            if len(point.shape) == 1:
                point = point.unsqueeze(0)
            point_label = torch.tensor([1])  # Foreground pixel

        return {'image': img, 'emb': emb, 'mask_bb': mask_bb.to(torch.long), 'mask_point': mask_point.to(torch.long),
                'point': point, 'point_label': point_label, 'boxes': bounding_box, 'ignore_index': self.ignore_index,
                'image_data': img_data, 'emb_data': emb_data, 'original_size': img.shape[-2:], 'mask': mask,
                'mask_downsampled': mask_downsampled}


class RetouchCirrus(RetouchDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, domain='cirrus', **kwargs)


class RetouchSpectralis(RetouchDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, domain='spectralis', **kwargs)


class RetouchTopcon(RetouchDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, domain='topcon', **kwargs)


class RetouchFourierCirrus(RetouchFourierDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, domain='cirrus', **kwargs)

    def __repr__(self):
        return self.__class__.__name__.replace('Fourier', '')


class RetouchFourierSpectralis(RetouchFourierDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, domain='spectralis', **kwargs)

    def __repr__(self):
        return self.__class__.__name__.replace('Fourier', '')


class RetouchFourierTopcon(RetouchFourierDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, domain='topcon', **kwargs)

    def __repr__(self):
        return self.__class__.__name__.replace('Fourier', '')


class Retouch:
    test_subset = None

    def __init__(self,
                 preprocess=None,
                 location=os.path.expanduser('~/data'),
                 emb_location=None,
                 batch_size=128,
                 num_workers=8,
                 source_set='spectralis',
                 target_set='cirrus',
                 transformations=None,
                 shuffle_training=True,
                 fold_number=0,
                 **kwargs):
        assert source_set in domains and target_set in domains
        args = kwargs['kwargs']
        self.source_set = source_set
        self.target_set = target_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_location = os.path.join(location, source_set)
        self.test_location = os.path.join(location, target_set)
        if args.__contains__("transformations_sample") and args.transformations_sample != 1:
            trans_sample = args.transformations_sample
        else:
            trans_sample = 1
        print("Loading Test Data from ", self.test_location)
        if emb_location is None:
            dataset_cls = RetouchDataset
            # dataset_cls = RetouchFourierDataset if 'fourier' in args.model else RetouchDataset

            if transformations is None and trans_sample == 0:
                train_transformations = v2.Compose([
                    GaussianNoise(mean=0, sigma=(1, 5)),
                    v2.RandomRotation(20),
                    v2.RandomHorizontalFlip(),
                    # v2.GaussianBlur(kernel_size=3),
                ])

                val_transformations = None
            else:
                # We want to do transformations for TTDA
                if args.task == 'test_time_volume':
                    dataset_cls = partial(RetouchDataset, augmenting_transforms=transformations,
                                          add_volume_information=True, max_batch_size=trans_sample)
                else:
                    dataset_cls = partial(RetouchDataset, augmenting_transforms=transformations)
                train_transformations = None
                val_transformations = None
            self.train_dataset = dataset_cls(root=location, domain=source_set, split='train', fold_number=fold_number,
                                             only_fluid=args.only_fluid, transforms=train_transformations,
                                             transformations_per_sample=trans_sample)
            self.val_dataset = dataset_cls(root=location, domain=target_set, split='val', fold_number=fold_number,
                                           only_fluid=args.only_fluid, transforms=val_transformations,
                                           transformations_per_sample=trans_sample)
            self.test_dataset = dataset_cls(root=location, domain=target_set, split='test', fold_number=fold_number,
                                            only_fluid=args.only_fluid, transforms=val_transformations,
                                            transformations_per_sample=trans_sample)

        else:
            self.train_dataset = RetouchEmbeddingsDataset(root=emb_location, root_base=location, domain=source_set,
                                                          split='train', only_fluid=args.only_fluid)
            self.val_dataset = RetouchEmbeddingsDataset(root=emb_location, root_base=location, domain=target_set,
                                                        split='val', only_fluid=args.only_fluid)

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
        v2.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0),
        v2.GaussianBlur(kernel_size=(5, 5)),
        v2.RandomAdjustSharpness(sharpness_factor=2)
    ])

    root_folder = '/Users/javier/Documents/datasets/Retouch/'
    dataset = RetouchDataset(root_folder, 'spectralis', 'train', transforms=trans,
                             only_fluid=False, augmenting_transforms=trans, fold_number=0)
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

        fix, ax = plt.subplots(1, 3)
        ax[0].imshow(torchvision.utils.draw_bounding_boxes(im, box, colors='red').permute(1, 2, 0))
        show_points(point, ax[0])
        ax[1].imshow(m_box[0])
        ax[2].imshow(m_point[0])

        ax[0].set_title('Original image')
        ax[1].set_title('Mask BB')
        ax[2].set_title('Mask Point')
        plt.show()
        break

        # Save to a file
        # plt.savefig(f'tests_mask/{i}.png')
