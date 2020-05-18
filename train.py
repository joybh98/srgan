import tensorflow as tf
from generator import generator
from discriminator import discriminator
from optimizers import generator_loss,discriminator_loss

EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 10

# seed is used as input for generator to generate data
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

def train(dataset,epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train()
