import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
import skimage.draw
from pathlib import Path
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# for augmentation pipeline
import imgaug as ia
import imgaug.augmenters as iaa
import tempfile


# data loader

class Lineament_dataloader(Dataset):
    #  no cusps
    label_dict = {"band": 1, "double_ridge": 2, "ridge_complex": 3, "undifferentiated_linea": 4}

    def __init__(self, dataset_path, transform=None, bbox_threshold=100):
        super(Lineament_dataloader, self).__init__()

        # this step also transforms to tensor
        if transform is not None:
            # if transform is None, do nothing,
            # if it's not None, check if it is a string
            if isinstance(transform, str):
                if transform == 'train_augs':
                    # if it is the keyword 'train_augs',
                    # then we apply the default albumentation augmentations defined in this class
                    self.transform = self.albu_augs()
                elif transform == 'simple_augs':
                    self.transform = self.simple_augs()
                elif transform == 'pub_augs1':
                    self.transform = self.pub_augs1()
                else:
                    # in any other case, set to None
                    self.transform = None
            else:
                self.transform = transform
        else:
            self.transform = None

        self.bbox_threshold = bbox_threshold

        # get list of image paths
        self.imgs = sorted(Path(dataset_path).glob(f"*.npy"))  # sorted(dataset_dir.glob(f"**/*.{fileid}"))

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]

        boxes = []
        # mask
        # it is stored in a geojson file
        try:
            mask_path = img_path.parents[0].joinpath(img_path.stem + ".json")  # self.masks[idx]
            # read in numpy array
            img_np = np.load(img_path)
            img = Image.fromarray(img_np * 255).convert("RGB")

            # open geojson
            with open(mask_path) as jsf:
                geoms = json.load(jsf)

            width, height = geoms['properties']['width'], geoms['properties']['height']

            # to mask:
            # WATCH OUT! we need to adjust npolyg to get rid of cusps
            npolyg = len([1 for frs in geoms['features'] if frs['properties']['id_int'] != 4])
            masks = np.zeros([height, width, npolyg], dtype=np.uint8)
            mask_ids = []

            # get rid of cusps
            # filter geoms['features']
            geomsfetsls = []
            for fetsr in geoms['features']:
                catstr = int(fetsr['properties']['id_int'])
                if catstr != 4:
                    # then we can append!
                    geomsfetsls.append(fetsr)

            for i, feature in enumerate(geomsfetsls):
                # get rid of cusps
                catstr = int(feature['properties']['id_int'])
                if catstr == 4:  # get rid of cusps
                    catstr = 6
                if catstr == 5:  # transform 5-> 4, otherwise we get cuda error
                    catstr = 4  # cast unidff lineae to category 4

                # check first if the data category is really defined in our class dict
                # if this is not the case, skip this feature
                if catstr not in self.label_dict.values():
                    continue

                # sometimes, a feature consists of multiple polygons!
                # therefore, we loop over these
                for fts_idx in range(len(feature['geometry']['coordinates'])):
                    p = feature['geometry']['coordinates'][fts_idx][0]
                    # convert to array
                    parr = np.asarray(p)
                    # convert coordinates to pixels
                    # p_conv = [(p[idx][0] - uly)/yres for idx in range(0, 1)]

                    parrx = parr[:, 0]
                    parry = parr[:, 1]

                    # Get indexes of pixels inside the polygon and set them to 1
                    # we get a new mask for every feature (index i)
                    rr, cc = skimage.draw.polygon(parry, parrx, masks.shape)
                    masks[rr, cc, i] = 1

                # for each polygon (= feature, I have one class_id)
                mask_ids.append(catstr)  # watch out for this to get rid of cusps

            # if I skipped everything, then there are no mask_ids
            if len(mask_ids) == 0:
                # this is kind of a hack
                # we throw a NotADirectoryError so that we end up directly in the 'except' statement
                raise NotADirectoryError('no masks found at all')

            # Return mask, and array of class IDs of each instance.
            mask_ids_arr = np.array(mask_ids, dtype=np.int32)

            #### augment now!
            # new: I located the augmentation pipeline right after the masks,
            # so that the bounding boxes get calculated only after, e.g., rotating by 45°,
            # which would pose problems otherwise
            # The pipeline must be in 'try', because it might happen that due to the bbox_threshold,
            # all masks and bounding boxes get deleted
            if self.transform is not None:
                # if with bounding boxes: masks_ex = np.moveaxis(np.expand_dims(np.array(masks), axis=0), 1,3) # N,H,W, num_masks
                masks_ex = np.expand_dims(np.array(masks), axis=0)
                image_ex = np.array(img)  # expects N,H,W,C as input
                # boxes_ex = np.expand_dims(np.array(boxes), axis=0) # expects input (N,B,4) (?), https://github.com/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb
                # print(image_ex.shape) # (80, 80, 3)
                # print(masks_ex.shape) # (1, 80, 80, 1)
                # print(boxes.shape) # torch.Size([1, 4])
                # print(boxes_ex.shape) # (1, 1, 4)
                img, masks = self.transform(image=image_ex,
                                            segmentation_maps=masks_ex)  # https://github.com/aleju/imgaug-doc/blob/master/notebooks/B05%20-%20Augment%20Segmentation%20Maps.ipynb
                # bounding_boxes=boxes_ex) # in what format are the bounding boxes?
                # print(img.shape) # (80, 80, 3)
                # print(masks.shape) # (1, 80, 80, 1)
                # convert back to tensor
                trans = transforms.ToTensor()
                img = trans(img.copy())
                masks = trans(masks[0]).permute(1, 2, 0)  # permuting necessary if not with bboxes
                # we need to transform the mask back to a 0/1 uint8 type mask
                masks = masks.bool().type(torch.uint8)
                # print(img.shape) # torch.Size([3, 80, 80])
                # print(masks.shape) # torch.Size([1, 80, 80])
                # boxes:
                # boxes = trans(boxes[0])[0]
                # print(boxes.shape) # torch.Size([1, 4])
                # plt.imshow(masks[:,:,0])
                # plt.show()

            # get bounding box coordinates for each mask
            num_objs = len(mask_ids_arr)
            boxes = []
            index_del = []
            for i in range(num_objs):
                pos = np.where(masks[:, :, i])
                # it may happen that after augmentation, one mask is empty
                # check this here
                # pos is a tuple with arrays for each dimension
                if len(pos[0]) != 0 or len(pos[1]) != 0:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])

                    # calculate pixel
                    A = abs((xmax - xmin) * (ymax - ymin))
                    # Check if area of mask is larger than a threshold
                    # print(A)
                    # e.g. if bbox_threshold=100, bounding boxes smaller than 10x10 get ignored.
                    if A < self.bbox_threshold:
                        # store index that needs to get deleted in a list
                        index_del.append(i)
                        continue

                    # only append boxes after area check
                    boxes.append([xmin, ymin, xmax, ymax])
                    # for albumentations: this is in pascal_voc format
                    # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
                # else:
                # print('empty masks i={}'.format(i))
                else:
                    # it might happen that pos is zero itself!
                    # store index that needs to get deleted in a list
                    # debugging
                    index_del.append(i)
                    # print('delete index')

            if len(index_del) > 0:
                # print('index gets deleted: {}'.format(index_del))
                # delete now masks
                masks = np.delete(masks, index_del, axis=2)
                # delete mask IDs
                mask_ids_arr = np.delete(mask_ids_arr, index_del)

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there are multiple classes
            labels = torch.as_tensor(mask_ids_arr, dtype=torch.int64)
            # masks as tensor
            masks = torch.as_tensor(masks, dtype=torch.uint8).permute(2, 0,
                                                                      1)  # here we need to change the order of the mask

            try:
                # here, boxes are already tensors
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            except IndexError:
                # check if there is at least one bbox, otherwise return an empty target
                # if isinstance(boxes, list):
                if len(boxes) == 0:
                    # print('no error, but no mask found for {}'.format(img_path))
                    # transform to tensor if not already happened
                    if not torch.is_tensor(img):
                        trans = transforms.ToTensor()
                        img = trans(img)
                    boxes, labels, masks, area, iscrowd = self.empty_target(img.shape)
                    # but now we do not want to check the

            # suppose all instances are not crowd
            # instances with iscrowd are ignored during evaluation
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # check again for all if there is at least one bbox, otherwise return an empty target
            # if isinstance(boxes, list):
            if len(boxes) == 0:
                # print('no error, but no mask found for {}'.format(img_path))
                if not torch.is_tensor(img):
                    trans = transforms.ToTensor()
                    img = trans(img)
                boxes, labels, masks, area, iscrowd = self.empty_target(img.shape)

        except NotADirectoryError:
            # then there is no mask!
            # TODO:
            # if wished by keyword,
            # I return an empty mask
            # get height and width of image to create empty mask in size of image

            # transform image to tensor so that they have the correct shape
            trans = transforms.ToTensor()
            img = trans(img)
            boxes, labels, masks, area, iscrowd = self.empty_target(img.shape)

        image_id = torch.tensor([idx])

        # for all cases:
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["path"] = img_path  # since all targets need to be sent to the device, this target is left out!

        if not torch.is_tensor(img):
            # else, we need to transform to a tensor at least!
            # print(type(img))
            trans = transforms.ToTensor()
            img = trans(img)

        return img, target

    def __len__(self):

        return len(self.imgs)

    def empty_target(self, img_shape):
        # get bounding box coordinates for each mask
        num_objs = 1

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor([[1.0, 1.0, 4.0, 4.0]],
                                dtype=torch.float32)  # [None] # add a dimension since model expects tensor of shape [N, 4]
        # there are multiple classes
        labels = torch.as_tensor([0], dtype=torch.int64)
        # masks as tensor
        masks = torch.as_tensor(np.zeros((1, img_shape[1], img_shape[2])), dtype=torch.uint8)

        area = torch.as_tensor([2], dtype=torch.uint8)
        # suppose all instances are not crowd
        # instances with iscrowd are ignored during evaluation
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        return boxes, labels, masks, area, iscrowd

    def albu_augs(self):
        #  implement augmentations
        # choose augmentations from imgaug https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        augmentation = iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images)

            sometimes(iaa.Affine(
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                # rotate=(-45, 45), # rotate by -45 to +45 degrees --> this is a bad idea with bounding boxes!
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           # sometimes( # iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(1, 5)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),

                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5)  # , # randomly remove up to 10% of the pixels
                               # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),

                           # iaa.Invert(0.05, per_channel=True), # not needed! (grayscale!) invert color channels

                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           # iaa.AddToHueAndSaturation((-100, 100)), # IT SEEMS THAT THIS CAUSES ERROR! # change hue and saturation

                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 2.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )  # , iaa.SaveDebugImageEveryNBatches(folder_path, 100)
        ], random_order=True)

        return augmentation

    def simple_augs(self):
        # #  implement augmentations
        # # choose augmentations from imgaug https://github.com/aleju/imgaug
        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        folder_path = Path('Z:/Groups/PIG/Caroline/lineament_detection/pytorch_maskrcnn/output/aug_imgs')
        # with tempfile.TemporaryDirectory() as folder_path:

        augmentation = iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images)
            iaa.ShearY((-20, 20)),  # (-20, 20)
            iaa.ShearX((-20, 20)),  # (-20, 20)
            iaa.Rotate((-45, 45)),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((1, 3),
                       [
                           iaa.OneOf([
                               # contrast
                               iaa.AddElementwise((-10, 10)),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1)),  # sharpen images
                               iaa.Add((-40, 40), per_channel=0),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.LinearContrast((0.5, 2.0), per_channel=0),  # improve or worsen the contrast
                               iaa.CLAHE(clip_limit=(3, 5)),
                               # This augmenter applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to images, a form of histogram equalization that normalizes within local image patches.
                           ]),
                           # iaa.Canny(alpha=0.2, colorizer=iaa.RandomColorsBinaryImageColorizer(color_true=255,color_false=0)), # (0.0, 0.5) canny edge detector blended with image
                           iaa.Invert(0.5),
                           iaa.CoarseDropout((0.0, 0.02), size_percent=(0.3, 0.5)),
                           # Drop 0 to 2% of all pixels by converting them to black pixels, but do that on a lower-resolution version of the image that has 30% to 50% of the original size, leading to large rectangular areas being dropped:
                           iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                       ],
                       random_order=True
                       ), iaa.SaveDebugImageEveryNBatches(folder_path, 100)
        ], random_order=True)

        return augmentation

    def pub_augs1(self):
        # use this abandoned augs as debug augs
        augmentation = iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        ])

        return augmentation
