import random
from datetime import datetime
from pathlib import Path

import albumentations
import cv2
import keras
import keras.models
import numpy as np
import pandas as pd
import segmentation_models.metrics
import segmentation_models.utils
import skimage.io
import skimage.transform
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau###, TensorBoard
from segmentation_models import Unet, get_preprocessing
from segmentation_models import losses as sm_losses

from tools.model import data_generator, input_preprocessing
from utils import nifti
from utils import image as image_utils


PROJECT_DIR = Path('../../../')
### DATA_DIR = PROJECT_PATH / 'data/ms-lesions/'
DATA_DIR = Path(r'C:\MyDiskBackup\Data\brain')
SERIES_DIR = DATA_DIR / 'series'
MASKS_DIR = DATA_DIR / 'masks'

DATA_CSV_DIR = PROJECT_DIR / 'data' / 'brain' / 'csv' / 'all'
TRAIN_DATA_CSV_PATH = DATA_CSV_DIR / 'train.csv'
VALID_DATA_CSV_PATH = DATA_CSV_DIR / 'valid.csv'

MODEL_ARCHITECTURE = Unet
BACKBONE = 'densenet201'
LOSS = sm_losses.bce_dice_loss
INPUT_SIZE = (256, 256)
BATCH_SIZE = 17
SMALL_DESCRIPTION = ''

BACKBONE_INPUT_PREPROCESSING = get_preprocessing(BACKBONE)

INPUT_CHANNELS = 3
CLASSES_NUMBER = 1

AUGMENTATIONS = albumentations.Compose(transforms=[
    albumentations.ShiftScaleRotate(
        border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.2, scale_limit=0.2, p=1),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    # RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1),
    # RandomGamma(p=1)  # (gamma_limit=(50, 150), p=1)
    ])

INPUT_PREPROCESSOR = input_preprocessing.InputPreprocessor(
    INPUT_SIZE, INPUT_CHANNELS, CLASSES_NUMBER, AUGMENTATIONS, BACKBONE_INPUT_PREPROCESSING)

DATE = datetime.now().strftime("%Y.%m.%d")
PARAMS_STR = f'{DATE}-Class{CLASSES_NUMBER}-{MODEL_ARCHITECTURE.__name__}-{BACKBONE}-{LOSS.__name__}-{INPUT_SIZE[0]}x{INPUT_SIZE[1]}-Batch{BATCH_SIZE}'
if SMALL_DESCRIPTION:
    PARAMS_STR += '-' + SMALL_DESCRIPTION
MODEL_NAME = PARAMS_STR + '.h5'

OUTPUT_DIR = PROJECT_DIR / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
MODEL_PATH = MODELS_DIR / MODEL_NAME
PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'


def central_image_mask_slices(series_path):
    series = nifti.read_image(series_path)
    slice_number = series.shape[2] // 2
    image_slice = series[..., slice_number]

    mask_path = series_path.parents[1] / 'Masks' / series_path.name
    mask = nifti.read_image(mask_path)
    mask_slice = mask[..., slice_number]

    return image_slice, mask_slice


def preprocessing_test(series_path: Path):
    image_slice, mask_slice = central_image_mask_slices(series_path)
    image, mask = preprocess_image_mask(image_slice, mask_slice)

    show_images((image_slice, mask_slice, image, mask))


def train():
    train_gen = data_generator.DataGenerator(
        INPUT_PREPROCESSOR, DATA_DIR, TRAIN_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=True)
    valid_gen = data_generator.DataGenerator(
        INPUT_PREPROCESSOR, DATA_DIR, VALID_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=False)

    # test
    '''
    print('len of train_gen', len(train_gen))
    train_batch = train_gen.__getitem__(0)
    train_series_batch, train_masks_batch = train_batch
    print('train_batch', train_series_batch.shape, train_masks_batch.shape)
    '''

    model = MODEL_ARCHITECTURE(backbone_name=BACKBONE, input_shape=(None, None, 3), classes=CLASSES_NUMBER,
                               encoder_weights='imagenet', encoder_freeze=True)

    dice_score = segmentation_models.metrics.f1_score   #####%! dice_score
###    dice_score.__name__ = 'dice_score'
    model.compile('Adam', loss=LOSS, metrics=[dice_score, segmentation_models.metrics.iou_score])
    model.summary()

    checkpoint = ModelCheckpoint(str(MODEL_PATH), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor="val_loss", patience=60, mode="min")
###    tensorboard_callback = TensorBoard(log_dir=str(MODELS_DIR / 'logs' / PARAMS_STR), write_graph=False)

    callbacks = [checkpoint, reduce_lr, early_stopping]###, tensorboard_callback]

    model.fit_generator(generator=train_gen,
                        epochs=100,
                        callbacks=callbacks,
                        validation_data=valid_gen)


def predict(model_path: Path = MODEL_PATH, number_of_batches_for_prediction: int = 2):
    test_gen = data_generator.DataGenerator(
        INPUT_PREPROCESSOR, DATA_DIR, VALID_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=False)

    model = keras.models.load_model(str(model_path), compile=False)

    save_dir = PREDICTIONS_DIR
    IMAGES_SAVE_DIR = save_dir / 'images'
    MASKS_SAVE_DIR = save_dir / 'masks'
    PREDICTED_MASKS_SAVE_DIR = save_dir / 'predictions'

    # Select random unique batches
    total_number_of_batches = len(test_gen)
    number_of_batches_for_prediction = min(number_of_batches_for_prediction, total_number_of_batches)
    random_batch_indexes = random.sample(range(total_number_of_batches), number_of_batches_for_prediction)

    image_id = 0
    for batch_index in random_batch_indexes:
        batch = test_gen.__getitem__(batch_index)
        series_batch, masks_batch = batch

        predicted_batch = model.predict(series_batch)

        # Save all images in batches
        for batch_image_index in range(series_batch.shape[0]):
            image = series_batch[batch_image_index]
            mask = masks_batch[batch_image_index]
            predicted_mask = predicted_batch[batch_image_index]

            image = image_utils.normalized_image(image)

            mask = np.stack((np.squeeze(mask),) * 3, axis=-1)
            predicted_mask = np.stack((np.squeeze(predicted_mask),) * 3, axis=-1)

            skimage.io.imsave(str(IMAGES_SAVE_DIR / f'{image_id}.png'), image)
            skimage.io.imsave(str(MASKS_SAVE_DIR / f'{image_id}.png'), mask)
            skimage.io.imsave(str(PREDICTED_MASKS_SAVE_DIR / f'{image_id}.png'), predicted_mask)

            image_id += 1


def generate_train_valid_csv(all_series_path: Path, masks_path: Path, train_part: float = 0.75):
    data_file_names = []
    for series_path in all_series_path.iterdir():
        mask_path = masks_path / series_path.name
        if mask_path.exists():
            data_file_names.append(series_path.name)
        else:
            print('WARNING: no mask', mask_path)

    train_file_names = random.sample(data_file_names, int(train_part * len(data_file_names)))
    valid_file_names = [file_name for file_name in data_file_names if file_name not in train_file_names]

    COLUMNS = ['file_names']
    train_data_frame = pd.DataFrame(data=train_file_names, columns=COLUMNS)
    train_data_frame.to_csv(str(TRAIN_DATA_CSV_PATH), index=False)

    valid_data_frame = pd.DataFrame(data=valid_file_names, columns=COLUMNS)
    valid_data_frame.to_csv(str(VALID_DATA_CSV_PATH), index=False)


def main():
    # generate_train_valid_csv(SERIES_PATH, MASKS_PATH)
    # train()
    predict(MODELS_DIR / '2020.02.11-Class1-Unet-densenet201-binary_crossentropy_plus_dice_loss-256x256-Batch17.h5')


if __name__ == '__main__':
    main()
