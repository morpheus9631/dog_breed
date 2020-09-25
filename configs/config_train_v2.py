# config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.WORK = CN()
_C.WORK.PATH = ""

_C.DATA = CN()
_C.DATA.PATH = ""
_C.DATA.PATH_TEST = ""
_C.DATA.PATH_TRAIN = ""
_C.DATA.FNAME_LABELS = ""

_C.PROCESSED = CN()
_C.PROCESSED.PATH  = ""
_C.PROCESSED.FNAME_BREEDS = ""
_C.PROCESSED.FNAME_LABELS = ""

_C.PRETRAINED = CN()
_C.PRETRAINED.PATH  = ""
_C.PRETRAINED.FNAME_PREMODEL = ""

_C.OUTPUT = CN()
_C.OUTPUT.PATH = ""

_C.TRAIN = CN()
_C.TRAIN.FRAC_FOR_TRAIN = 0.8
_C.TRAIN.NUM_CLASSES = 16
_C.TRAIN.BATCH_SIZE = 16
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