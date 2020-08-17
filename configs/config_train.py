# config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.WORK = CN()
_C.WORK.ROOT_PATH = "D:\\GitWork\\dog_breed\\"
_C.WORK.LABELS_FNAME = "labels.csv"

_C.DATA = CN()
_C.DATA.ROOT_PATH = "D:\\Dataset\\dog_breed\\"
_C.DATA.TEST_DIR = "test"
_C.DATA.TRAIN_DIR = "train"
_C.DATA.FRAC_FOR_TRAIN = 0.8


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()