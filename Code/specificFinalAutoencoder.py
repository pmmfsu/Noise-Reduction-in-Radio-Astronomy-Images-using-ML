import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '10'

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
import time
import numpy as np
from tf_fits.image import image_decode_fits
from tensorflow.keras import layers


class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(256, 256, 1)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(8, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.MaxPooling2D((2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(16, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.MaxPooling2D((2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(32, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.MaxPooling2D((2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(64, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.MaxPooling2D((2, 2))

        ])
        self.decoder = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(64, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.UpSampling2D(size=(2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(32, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.UpSampling2D(size=(2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(16, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.UpSampling2D(size=(2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(8, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.UpSampling2D(size=(2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(1, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(1, (3, 3), padding='valid', activation=tf.nn.sigmoid),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_clean_path(path):
    file_name = os.path.basename(path)
    file_name_split = file_name.split("_")
    if file_name_split[0] == "clean":
        return path, path
    elif file_name_split[1] == "poisson":
        pathSplit = path.split('/')
        cleanPath = ""
        for i in range(len(pathSplit) - 3):
            cleanPath += pathSplit[i] + "/"

        cleanPath += 'clean_data_' + file_name_split[-2] + '_' + file_name_split[-1]
        return path, cleanPath
    else:
        pathSplit = path.split('/')
        # cleanPath = pathSplit[0] + '/' + pathSplit[1] + '/' + pathSplit[2] + '/' + 'clean_data_' + file_name_split[-2] + '_' + file_name_split[-1]
        cleanPath = ""
        for i in range(len(pathSplit) - 4):
            cleanPath += pathSplit[i] + "/"

        cleanPath += 'clean_data_' + file_name_split[-2] + '_' + file_name_split[-1]
        return path, cleanPath


def get_files(path):
    fits_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".fits"):
                variables = file.split('_')
                if variables[0] == 'noise':
                    if variables[1] == 'poisson': #gaussian impulse poisson
                        fits_files.append(get_clean_path(os.path.join(root, file)))
    return fits_files


@tf.function(reduce_retracing=True)
def processing_map(input_images, output_images):
    input_data = tf.expand_dims(tf.clip_by_value(tf.divide(tf.add(input_images, 0), 100000.0), 0., 1.), axis=-1)
    output_data = tf.expand_dims(tf.clip_by_value(tf.divide(tf.add(output_images, 0), 100000.0), 0., 1.), axis=-1)
    return input_data, output_data


@tf.function(reduce_retracing=True)
def open_files_batch(path_batch):
    input_data_batch = tf.map_fn(lambda x: tf.io.read_file(x[0]), path_batch, fn_output_signature=tf.string)
    output_data_batch = tf.map_fn(lambda x: tf.io.read_file(x[1]), path_batch, fn_output_signature=tf.string)
    return input_data_batch, output_data_batch


@tf.function(reduce_retracing=True)
def read_files_batch(input_binary_strings_batch, output_binary_strings_batch):
    input_data_batch = tf.map_fn(lambda x: image_decode_fits(x, 0), input_binary_strings_batch,
                                 fn_output_signature=tf.float32)
    output_data_batch = tf.map_fn(lambda x: image_decode_fits(x, 0), output_binary_strings_batch,
                                  fn_output_signature=tf.float32)
    return input_data_batch, output_data_batch


print("Get Dataset Paths", end='')
start_time = time.time()
path = "dataset"
file_list = get_files(path)
file_list = np.array(file_list)
np.random.shuffle(file_list)
end_time = time.time()
print(f": Took {end_time - start_time} seconds")

train_files = file_list[:int(len(file_list) * 0.8)]
val_files = file_list[int(len(file_list) * 0.8):]

train_dataset = tf.data.Dataset.from_tensor_slices(train_files) \
    .shuffle(buffer_size=100000, seed=1213, reshuffle_each_iteration=True) \
    .batch(16 * 4) \
    .map(open_files_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .map(read_files_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .map(processing_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(val_files) \
    .batch(16 * 4) \
    .map(open_files_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .map(read_files_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .map(processing_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

strategy = tf.distribute.MirroredStrategy()
print("poisson Autoencoder")
with strategy.scope():
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(jit_compile=False),
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=[tf.keras.metrics.MeanSquaredLogarithmicError(), tf.keras.metrics.MeanAbsolutePercentageError()])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)
autoencoder.fit(train_dataset,
                epochs=50,
                verbose=2,
                validation_data=val_dataset,
                use_multiprocessing=True,
                workers=20,
                max_queue_size=1, callbacks=[early_stopping]
                )

autoencoder.save('/home/pmmf/MastersCode/autoencoderModelsv2/PoissonAutoencoder')

