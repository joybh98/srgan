from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

config.TRAIN.batch_size = 1000
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9
# initialize G
config.TRAIN.n_epoch_init = 10

# SRGAN
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

# TRAIN set location
config.TRAIN.hr_img_path = '/home/joy/datasets/64x/train/'
config.TRAIN.lr_img_path = '/home/joy/datasets/32x/train/'

config.TEST = edict()
# TEST set location
config.TEST.hr_img_path = '/home/joy/datasets/64x/test/'
config.TEST.lr_img_path = '/home/joy/datasets/32x/test/'

config.VALID = edict()


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
