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
_C.PROCESSED.FNAME_BREEDS = "breeds_processed"
_C.PROCESSED.FNAME_LABELS = "labels_processed"

_C.PRETRAINED = CN()
_C.PRETRAINED.PATH  = "D:\\GitWork\\dog_breed\\models\\"
_C.PRETRAINED.FNAME_RESNET18 = 'resnet18-5c106cde.pth'
_C.PRETRAINED.FNAME_RESNET50 = 'resnet50-19c8e357.pth'
_C.PRETRAINED.FNAME_INCEPTION_V3 = 'inception_v3-1a9a5a14.pth'

_C.TRAIN = CN()
_C.TRAIN.FRAC_FOR_TRAIN = 0.8
_C.TRAIN.NUM_CLASSES = 16
_C.TRAIN.BATCH_SIZE = 6
_C.TRAIN.LEARNING_RATE = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.STEP_SIZE = 7
_C.TRAIN.GAMMA = 0.1
_C.TRAIN.NUM_EPOCHS = 2


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()