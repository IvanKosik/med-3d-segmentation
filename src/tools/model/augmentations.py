import random

import cv2
from albumentations import PadIfNeeded, Resize, DualTransform, IAACropAndPad


import imgaug as ia

try:
    from imgaug import augmenters as iaa
except ImportError:
    import imgaug.imgaug.augmenters as iaa


class ElasticSize2(DualTransform):
    def __init__(self, pad_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        self.pad_limit = pad_limit
        self.border_mode = border_mode
        self.value = value

#        self._pad_transform = PadIfNeeded(border_mode=self.border_mode, value=self.value)

    def apply(self, img, pad_side=0, pad_factor=0, **params):
        return self._transformed(img, pad_side, pad_factor)

    def apply_to_mask(self, img, pad_side=0, pad_factor=0, **params):
        return self._transformed(img, pad_side, pad_factor)

    def _transformed(self, img, pad_side=0, pad_factor=0):
        # side_percent_ranges = [(0, 0)] * 4

        side_percents = [0] * 4
        side_percents[pad_side] = pad_factor

        w, h = img.shape[:2]

        aug = iaa.CropAndPad(percent=tuple(side_percents), pad_cval=self.value)
        img = aug.augment_image(img)

        # aug = iaa.CropAndPad(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

        # aug = iaa.CropAndPad(percent=((0, self.pad_limit), (0, self.pad_limit), (0, self.pad_limit), (0, pad_limit)))


        # pad_width = round(pad_factor * w)

#        img = self._pad_transform.apply(img, **{pad_side: pad_width})

        resize = Resize(w, h)
        return resize.apply(img)

    def get_params(self):
        return {
            'pad_side': random.choice([0, 1, 2, 3]),
            'pad_factor': random.uniform(0, self.pad_limit),  #-self.pad_limit, self.pad_limit),
        }

    def get_transform_init_args(self):
        return {
            'pad_limit': self.pad_limit,
            'border_mode': self.border_mode,
            'value': self.value,
        }



class ElasticSize(DualTransform):
    def __init__(self, pad_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        self.pad_limit = pad_limit
        self.border_mode = border_mode
        self.value = value

        self._pad_transform = PadIfNeeded(border_mode=self.border_mode, value=self.value)

    def apply(self, img, pad_side='pad_left', pad_factor=0, **params):
        return self._transformed(img, pad_side, pad_factor)

    def apply_to_mask(self, img, pad_side='pad_left', pad_factor=0, **params):
        return self._transformed(img, pad_side, pad_factor)

    def _transformed(self, img, pad_side='pad_left', pad_factor=0):


        # aug = iaa.CropAndPad(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))
        # aug = iaa.CropAndPad(percent=((0, self.pad_limit), (0, self.pad_limit), (0, self.pad_limit), (0, pad_limit)))


        w, h = img.shape[:2]
        pad_width = round(pad_factor * w)

        img = self._pad_transform.apply(img, **{pad_side: pad_width})

        resize = Resize(w, h)
        return resize.apply(img)

    def get_params(self):
        return {
            'pad_side': random.choice(['pad_left', 'pad_right', 'pad_bottom', 'pad_top']),
            'pad_factor': random.uniform(0, self.pad_limit),
        }

    def get_transform_init_args(self):
        return {
            'pad_limit': self.pad_limit,
            'border_mode': self.border_mode,
            'value': self.value,
        }


'''
def elastic_size(image, **kwargs):
    pad_limit = 0.3

    w, h = image.shape[:2]

    pad = PadIfNeeded(border_mode=cv2.BORDER_CONSTANT)
    if random.random() < 0.5:
        # Pad width
        pad_width = round(random.uniform(0, pad_limit * w))
        if random.random() < 0.5:
            image = pad.apply(image, pad_left=pad_width)
        else:
            image = pad.apply(image, pad_right=pad_width)
    else:
        # Pad height
        pad_height = round(random.uniform(0, pad_limit * h))
        if random.random() < 0.5:
            image = pad.apply(image, pad_bottom=pad_height)
        else:
            image = pad.apply(image, pad_top=pad_height)

    resize = Resize(w, h)
    return resize.apply(image)
'''
