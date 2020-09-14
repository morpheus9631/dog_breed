# config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.WORK = CN()
_C.WORK.PATH = "D:\\GitWork\\dog_breed\\"

_C.DATA = CN()
_C.DATA.PATH = "D:\\GitWork\\dog_breed\\data\\"
_C.DATA.CSV_LABELS = "labels.csv"
_C.DATA.CSV_SAMPLE_SUBMISSION = "sample_submission.csv"

_C.DATA.PATH_RAW  = "D:\\GitWork\\dog_breed\\data\\raw\\"
_C.DATA.DIR_TEST = "test"
_C.DATA.DIR_TRAIN = "train"

_C.PROCESSED = CN()
_C.PROCESSED.PATH  = "D:\\GitWork\\dog_breed\\data\\processed\\"
_C.PROCESSED.CSV_BREEDS = "breeds_processed.csv"
_C.PROCESSED.CSV_LABELS = "labels_processed.csv"
_C.PROCESSED.TRAIN_DATA_FILE = "train_data.npz"
_C.PROCESSED.VALID_DATA_FILE = "valid_data.npz"

_C.PRETRAINED = CN()
_C.PRETRAINED.PATH  = "D:\\GitWork\\dog_breed\\models\\"
_C.PRETRAINED.FNAME = 'resnet50-19c8e357.pth'

_C.TRAIN = CN()
_C.TRAIN.FRAC_FOR_TRAIN = 0.8
_C.TRAIN.NUM_POPULAR_CLASSES = 16
_C.TRAIN.BATCH_SIZE = 6
_C.TRAIN.LEARNING_RATE = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NUM_EPOCHS = 2


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()