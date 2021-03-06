import random
from datetime import datetime
from pathlib import Path
import re

import keras
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
from segmentation_models import get_preprocessing
from segmentation_models import losses as sm_losses
from segmentation_models import metrics as sm_metrics

from tools.model import data_generator, input_preprocessing, configs
from utils import image as image_utils
from utils import nifti, debug


BACKBONE_INPUT_PREPROCESSING = get_preprocessing(configs.BACKBONE)

INPUT_PREPROCESSOR = input_preprocessing.InputPreprocessor(
    configs.INPUT_SIZE, configs.INPUT_CHANNELS, configs.CLASSES_NUMBER,
    configs.AUGMENTATIONS, BACKBONE_INPUT_PREPROCESSING)

DATE = datetime.now().strftime("%Y.%m.%d")
PARAMS_STR = f'{DATE}-Class{configs.CLASSES_NUMBER}-{configs.MODEL_ARCHITECTURE.__name__}' \
             f'-{configs.BACKBONE}-{configs.LOSS.__name__}-{configs.INPUT_SIZE[0]}x{configs.INPUT_SIZE[1]}' \
             f'-Batch{configs.BATCH_SIZE}'
if configs.SMALL_DESCRIPTION:
    PARAMS_STR += '-' + configs.SMALL_DESCRIPTION
MODEL_NAME = PARAMS_STR + '.h5'

MODEL_PATH = configs.MODELS_DIR / MODEL_NAME


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
    train_gen = data_generator.DataGenerator(INPUT_PREPROCESSOR, configs.DATA_DIR, configs.TRAIN_DATA_CSV_PATH,
                                             batch_size=configs.BATCH_SIZE, is_train=True,
                                             skip_slices_with_empty_mask=configs.SKIP_SLICES_WITH_EMPTY_MASK)
    valid_gen = data_generator.DataGenerator(INPUT_PREPROCESSOR, configs.DATA_DIR, configs.VALID_DATA_CSV_PATH,
                                             batch_size=configs.BATCH_SIZE, is_train=False,
                                             skip_slices_with_empty_mask=configs.SKIP_SLICES_WITH_EMPTY_MASK)

    # test
    '''
    print('len of train_gen', len(train_gen))
    train_batch = train_gen.__getitem__(0)
    train_series_batch, train_masks_batch = train_batch
    print('train_batch', train_series_batch.shape, train_masks_batch.shape)
    '''

    model = configs.MODEL_ARCHITECTURE(backbone_name=configs.BACKBONE, input_shape=(None, None, 3),
                                       classes=configs.CLASSES_NUMBER, encoder_weights='imagenet', encoder_freeze=True)

    model.compile(keras.optimizers.Adam(learning_rate=1.3e-3), loss=configs.LOSS,
                  metrics=[sm_losses.binary_crossentropy, sm_losses.JaccardLoss(per_image=True),
                           sm_metrics.IOUScore(threshold=0.5, per_image=True)])
    model.summary(line_length=150)

    monitored_quantity_name = 'val_' + sm_metrics.iou_score.name
    monitored_quantity_mode = 'max'
    checkpoint = keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor=monitored_quantity_name, verbose=1,
                                                 save_best_only=True, mode=monitored_quantity_mode)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitored_quantity_name, factor=0.8, patience=6, verbose=1,
                                                  mode=monitored_quantity_mode, min_lr=1e-6)
    early_stopping = keras.callbacks.EarlyStopping(monitor=monitored_quantity_name, patience=60,
                                                   mode=monitored_quantity_mode)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(configs.MODELS_DIR / 'logs' / PARAMS_STR),
                                                       write_graph=False)

    callbacks = [checkpoint, reduce_lr, early_stopping, tensorboard_callback]

    model.fit_generator(generator=train_gen,
                        epochs=configs.EPOCHS,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=valid_gen)


def test_data_generator_with_prediction(model_path: Path = Path(), data_csv_path: Path = configs.VALID_DATA_CSV_PATH,
                                        number_of_batches_for_prediction: int = 2,
                                        save_dir: Path = configs.PREDICTIONS_DIR,
                                        skip_slices_with_empty_mask: bool = False):
    # If |model_path| defined, then we have to do a prediction, so use false for |is_train|, to disable augmentations
    is_train = not model_path.is_file()

    test_gen = data_generator.DataGenerator(INPUT_PREPROCESSOR, configs.DATA_DIR, data_csv_path,
                                            batch_size=configs.BATCH_SIZE, is_train=is_train,
                                            skip_slices_with_empty_mask=skip_slices_with_empty_mask)

    model = keras.models.load_model(str(model_path), compile=False) if model_path.is_file() else None

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

        predicted_batch = model.predict(series_batch) if model is not None else None

        # Save all images in batches
        for batch_image_index in range(series_batch.shape[0]):
            image = series_batch[batch_image_index]
            mask = masks_batch[batch_image_index]

            image = image_utils.normalized_image(image)
            mask = np.stack((np.squeeze(mask),) * 3, axis=-1)

            skimage.io.imsave(str(IMAGES_SAVE_DIR / f'{image_id}.png'), image)
            skimage.io.imsave(str(MASKS_SAVE_DIR / f'{image_id}.png'), mask)

            if predicted_batch is not None:
                predicted_mask = predicted_batch[batch_image_index]
                predicted_mask = np.stack((np.squeeze(predicted_mask),) * 3, axis=-1)
                skimage.io.imsave(str(PREDICTED_MASKS_SAVE_DIR / f'{image_id}.png'), predicted_mask)

            image_id += 1


def generate_train_valid_csv(all_series_dir: Path, masks_dir: Path,
                             train_csv_path: Path, valid_csv_path: Path,
                             filter_predicate=lambda file_name: True, train_part: float = 0.75):
    data_file_names = []
    for series_path in all_series_dir.iterdir():
        if not filter_predicate(series_path.name):
            continue

        mask_path = masks_dir / series_path.name
        if mask_path.exists():
            data_file_names.append(series_path.name)
        else:
            print('WARNING: no mask', mask_path)

    train_file_names = random.sample(data_file_names, int(train_part * len(data_file_names)))
    valid_file_names = [file_name for file_name in data_file_names if file_name not in train_file_names]

    COLUMNS = ['file_names']
    train_data_frame = pd.DataFrame(data=train_file_names, columns=COLUMNS)
    train_data_frame.to_csv(str(train_csv_path), index=False)

    valid_data_frame = pd.DataFrame(data=valid_file_names, columns=COLUMNS)
    valid_data_frame.to_csv(str(valid_csv_path), index=False)


def test_on_image():
    image = skimage.io.imread(r'D:\Temp\model_rotate_test\rotate\slices0png.png', as_gray=True)

    batch_image = np.zeros((configs.BATCH_SIZE, *configs.INPUT_SIZE, configs.INPUT_CHANNELS))

    image = INPUT_PREPROCESSOR.preprocess_image(image)[0]
    image = image * 255
    image = np.stack((image,) * INPUT_PREPROCESSOR.image_input_channels, axis=-1)

    batch_image[0, ...] = image
    batch_image = INPUT_PREPROCESSOR.backbone_input_preprocessing(batch_image)

    model = keras.models.load_model(r'D:\Projects\med-3d-segmentation\output\models\paranasal-sinuses\2020.05.15-Class1-Unet-densenet201-binary_crossentropy_plus_jaccard_loss-352x352-Batch8-BothClasses-Empty_1-Rotate90.h5', compile=False)
    predicted_batch = model.predict(batch_image)

    predicted_mask = predicted_batch[0]
    predicted_mask = np.stack((np.squeeze(predicted_mask),) * 3, axis=-1)
    skimage.io.imsave(r'D:\Temp\model_rotate_test\rotate\slices0png_PREDICTED.png', predicted_mask * 255)


def main():
    # generate_train_valid_csv(
    #     configs.SERIES_DIR, configs.MASKS_DIR, configs.TRAIN_DATA_CSV_PATH, configs.VALID_DATA_CSV_PATH)
        # filter_predicate=lambda file_name: re.search('[ _]t2.*tse', file_name, re.IGNORECASE) is not None)

    # To test only data generator
    # test_data_generator_with_prediction(number_of_batches_for_prediction=10)

    train()
    # test_on_image()

    # To test data generator and predictions
    # test_data_generator_with_prediction(
    #     configs.MODELS_DIR / '2020.04.04-Class1-Unet-densenet201-binary_focal_loss_plus_jaccard_loss-352x352-Batch8-BothClasses-Empty_1.h5',
    #     # MODEL_PATH,
    #     configs.VALID_DATA_CSV_PATH, number_of_batches_for_prediction=30)


if __name__ == '__main__':
    main()
