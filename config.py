from easydict import Easydict as edict
import json

config = edict()
config.TRAIN = edict()

# initialize G
config.train.n_epoch_init = 10

# SRGAN
config.TRAIN.n_epoch = 100

# TRAIN set location
# config.TRAIN.hr_img_path
# config.TRAIN.lr_img_path
# TEST set location
# config.TRAIN.hr_img_path
# config.TRAIN.lr_img_path

config.VALID = edict()


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
