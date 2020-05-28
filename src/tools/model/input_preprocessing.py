import numpy as np
import skimage.io
import skimage.transform

from utils import image as image_utils


class InputPreprocessor:
    def __init__(self, image_input_size, image_input_channels,
                 mask_classes_amount, augmentations, backbone_input_preprocessing):
        self.image_input_size = image_input_size
        self.image_input_channels = image_input_channels
        self.mask_classes_amount = mask_classes_amount
        self.augmentations = augmentations
        self.backbone_input_preprocessing = backbone_input_preprocessing

    def preprocess_image(self, image, is_mask: bool = False, axis_pads: tuple = None):
        '''
        if axis_pads is None:
            # Add zero-paddings to make square image (original image will be in the center of paddings)
            max_shape = max(image.shape)
            pad_sizes = max_shape - np.array(image.shape)

            axis_before_pads = np.rint(pad_sizes / 2).astype(np.int)
            axis_after_pads = pad_sizes - axis_before_pads

            axis_pads = tuple(zip(axis_before_pads, axis_after_pads))

        image = np.pad(image, axis_pads, mode='constant')
        '''

        image = skimage.transform.resize(image, self.image_input_size, order=3, anti_aliasing=True)  # preserve_range=True)

        # Normalize to [0, 1]
        if not is_mask:
            image = image_utils.normalized_image(image)

        return image, axis_pads

    def preprocess_image_mask(self, image, mask):
        image, axis_pads = self.preprocess_image(image)
        mask = self.preprocess_image(mask, is_mask=True, axis_pads=axis_pads)[0]
        # assert image.shape == mask.shape == INPUT_SIZE, 'image or mask has wrong size'
        return image, mask

    def augmentate_image_mask(self, image, mask):
        augmented = self.augmentations(image=image, mask=mask)
        return augmented['image'], augmented['mask']
