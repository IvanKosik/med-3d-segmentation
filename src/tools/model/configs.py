from pathlib import Path

import albumentations
import cv2
from segmentation_models import Unet
from segmentation_models import losses as sm_losses

from tools.model.augmentations import ElasticSize


PROJECT_DIR = Path('../../../')
OBJECT_NAME = 'paranasal-sinuses' #'ms-lesions'
### DATA_DIR = PROJECT_DIR / 'data' / OBJECT_NAME
DATA_DIR = Path(r'C:\MyDiskBackup\Data') / OBJECT_NAME
SERIES_DIR = DATA_DIR / 'series'
MASKS_DIR = DATA_DIR / 'masks'

DATA_CSV_DIR = PROJECT_DIR / 'data' / OBJECT_NAME / 'csv' / 'all'
TRAIN_DATA_CSV_PATH = DATA_CSV_DIR / 'train.csv'
VALID_DATA_CSV_PATH = DATA_CSV_DIR / 'valid.csv'
TEST_DATA_CSV_PATH = DATA_CSV_DIR / 'test.csv'

MODEL_ARCHITECTURE = Unet
BACKBONE = 'densenet201'
LOSS = sm_losses.bce_jaccard_loss
INPUT_SIZE = (352, 352)
BATCH_SIZE = 8
EPOCHS = 150
SMALL_DESCRIPTION = 'Air'

INPUT_CHANNELS = 3
CLASSES_NUMBER = 1

SKIP_SLICES_WITH_EMPTY_MASK = True

AUGMENTATIONS = albumentations.Compose(transforms=[
    albumentations.ShiftScaleRotate(
        border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.2, scale_limit=0.2, p=1),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomRotate90(p=1),
    ElasticSize(p=0.5),

    # RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1),
    # RandomGamma(p=1)  # (gamma_limit=(50, 150), p=1)
])

OUTPUT_DIR = PROJECT_DIR / 'output'
MODELS_DIR = OUTPUT_DIR / 'models' / OBJECT_NAME
PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
