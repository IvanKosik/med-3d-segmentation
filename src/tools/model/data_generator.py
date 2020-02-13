from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from keras.utils import Sequence

from utils import image as image_utils
from utils import nifti, debug


class DataGenerator(Sequence):
    def __init__(self, input_preprocessor: InputPreprocessor,
                 data_path: Path, data_csv_path: Path, batch_size, is_train):
        self.input_preprocessor = input_preprocessor
        self.batch_size = batch_size
        self.is_train = is_train
        self.discard_last_incomplete_batch = True

        self.series_path = data_path / 'series'
        self.masks_path = data_path / 'masks'

        self.images = []
        self.masks = []
        self.samples_amount = 0

        data_csv = pd.read_csv(str(data_csv_path))

        # Read all images/masks into memory to speed up training
        number_of_data_csv_rows = len(data_csv)
        for index, train_csv_row in enumerate(data_csv.values):
            nifti_name = train_csv_row[0]
            print(f'{index + 1}/{number_of_data_csv_rows}   {nifti_name}')

            nifti_series_path = self.series_path / nifti_name
            series = nifti.read_image(nifti_series_path)

            mask = nifti.read_image(self.masks_path / nifti_name)
            mask[mask != 1] = 0                                       ################# FOR 1 CLASS

            for slice_number in range(series.shape[2]):
                series_slice = series[..., slice_number]
                mask_slice = mask[..., slice_number]
                preprocessed_image, preprocessed_mask = self.input_preprocessor.preprocess_image_mask(
                    series_slice, mask_slice)
                self.images.append(preprocessed_image)
                self.masks.append(preprocessed_mask)

        self.samples_amount = len(self.images)

        self.images = np.array(self.images, dtype=np.float32)
        self.masks = np.array(self.masks, dtype=np.float32)

        debug.print_info(self.images, 'images')
        debug.print_info(self.masks, 'masks')

        self.sample_indexes = np.arange(self.samples_amount)
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch"""
        number_of_batches = self.samples_amount / self.batch_size
        return math.floor(number_of_batches) if self.discard_last_incomplete_batch else math.ceil(number_of_batches)

    def __getitem__(self, batch_index):
        """Generate one batch of data"""
        batch_image = np.zeros((self.batch_size, *self.input_preprocessor.image_input_size,
                                self.input_preprocessor.image_input_channels))
        batch_mask = np.zeros((self.batch_size, *self.input_preprocessor.image_input_size,
                               self.input_preprocessor.mask_classes_amount))

        # Generate sample indexes of the batch
        batch_sample_indexes = self.sample_indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        # Augmentate images
        for item_number, batch_sample_index in enumerate(batch_sample_indexes):
            image = self.images[batch_sample_index]
            mask = self.masks[batch_sample_index]

            if self.is_train:
                image, mask = self.input_preprocessor.augmentate_image_mask(image, mask)

                # Normalize once again image to [0, 1] after augmentation
                image = image_utils.normalized_image(image)

            image = image * 255
            image = np.stack((image,) * self.input_preprocessor.image_input_channels, axis=-1)
            batch_image[item_number, ...] = image

            batch_mask[item_number, ..., 0] = mask

            # batch_mask[item_number, ..., 0][np.where(augmented_mask == 0)] = 1
            # batch_mask[item_number, ..., 1][np.where(augmented_mask == 1)] = 1
            # batch_mask[item_number, ..., 2][np.where(augmented_mask == 6)] = 1

        batch_image = self.input_preprocessor.backbone_input_preprocessing(batch_image)
        return batch_image, batch_mask

    def on_epoch_end(self):
        """Shuffle files after each epoch"""
        if self.is_train:
            np.random.shuffle(self.sample_indexes)
