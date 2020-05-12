import os
import time
import tensorflow as tf
import tensorlayer as tl
from generator import generator
from discriminator import discriminator
from config import config
import multiprocessing
""" Hyperparameters """

## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128

# create folders to save result images and trained models
save_dir = "samples"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)


""" get_train_data()
This method takes the dataset as parameter and:
1. Loads it
2. Generator for image
3. Maps the input(image) & apply preprocessing
"""

def get_data():
    # Load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))#[0:20]
    # Read Images
    train_hr_imgs = tl.vis.read_images(train_hr_img_list,path=config.TRAIN.hr_img_path,n_threads=32)

    # Create generator for imgs; generator only iterates once and instead of return we use yield \
    # and they do not store all values in memory

    def generator_train():
        for img in train_hr_imgs:
            yield img
    
    # Map images and perform preprocessing functions on it
    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [384, 384, 3])
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=[96, 96])
        return lr_patch, hr_patch
    
    train_ds = tf.data.Dataset.from_generator(generator_train,output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
        # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    # train_ds = train_ds.shuffle(shuffle_buffer_size)
    # train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
    return train_ds


def train():
    G = generator((batch_size, 96, 96, 3))
    D = discriminator((batch_size, 384, 384, 3))
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

    lr_v = tf.Variable(lr_init)
    # g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    # g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    # d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    VGG.train()

    train_ds = get_data()

     ## initialize learning (G)
    n_step_epoch = round(n_epoch_init // batch_size)
    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))

    ## adversarial learning (G, D)
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_patchs+1)/2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))

        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g.h5'))
            D.save_weights(os.path.join(checkpoint_dir, 'd.h5'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")