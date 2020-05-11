from easydict import Easydict as edict
import json

config = edict()
config.TRAIN = edict()

# initialize G
config.train.n_epoch_init = 10

# SRGAN
config.TRAIN.n_epoch = 100

# TRAIN set location
config.TRAIN.hr_img_path = '/home/joy/datasets/64x/train/'
config.TRAIN.lr_img_path = '/home/joy/datasets/32x/train/'
# TEST set location
config.TEST.hr_img_path = '/home/joy/datasets/64x/test/'
config.TEST.lr_img_path = '/home/joy/datasets/32x/test/'

config.VALID = edict()


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
