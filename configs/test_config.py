import os, sys
from os.path import dirname

root_path = os.path.abspath(dirname(dirname(__file__)))
if not root_path in sys.path: sys.path.insert(0, root_path)

import argparse
from configs.config_train import get_cfg_defaults

def parse_args(**args):
    parser = argparse.ArgumentParser(description='dog breed')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml", help="Configurations.")
#     return parser.parse_args(**args)          # for python
    return parser.parse_known_args(**args)    # for jupyter notebook


def main():
    # arguments 
    args, _ = parse_args()
    print(args)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    abspath_label = os.path.join(cfg.WORK.ROOT_PATH, cfg.WORK.LABELS_FNAME)
    print(abspath_label)

if __name__ == '__main__':
    main()