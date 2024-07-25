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


def getLabel(path):  # 0 = clean, 1 = Gaussian, 2 = Impulse, 3 = Poisson
    file_name = os.path.basename(path)
    file_name_split = file_name.split("_")
    if file_name_split[0] == "clean":
        return path, 0
    elif file_name_split[1] == "gaussian":
        return path, 1
    elif file_name_split[1] == "impulse":
        return path, 2
    else:
        return path, 3


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Input(shape=(256, 256, 1)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(8, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.MaxPooling2D((2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(16, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.MaxPooling2D((2, 2)),
            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')),
            layers.Conv2D(32, (3, 3), padding='valid', activation=tf.nn.leaky_relu),
            layers.Flatten(),
            layers.Dense(64, activation=tf.nn.leaky_relu),
            layers.Dense(4, activation=tf.nn.softmax)
        ])

    def call(self, x):
        model = self.model(x)
        return model


def get_files(path):
    fits_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".fits"):
                fits_files.append(getLabel(os.path.join(root, file)))
    return fits_files


@tf.function(reduce_retracing=True)
def open_files_batch(path_batch):
    input_data_batch = tf.map_fn(lambda x: tf.io.read_file(x[0]), path_batch, fn_output_signature=tf.string)
    output_data_batch = tf.map_fn(lambda x: tf.strings.to_number(x[1], out_type=tf.int32), path_batch,
                                  fn_output_signature=tf.int32)
    return input_data_batch, output_data_batch


@tf.function(reduce_retracing=True)
def read_files_batch(input_binary_strings_batch, output_labels_batch):
    input_data_batch = tf.map_fn(lambda x: image_decode_fits(x, 0), input_binary_strings_batch,
                                 fn_output_signature=tf.float32)
    return input_data_batch, output_labels_batch


@tf.function(reduce_retracing=True)
def processing_map(input_images, output_label):
    input_data = tf.expand_dims(tf.clip_by_value(tf.divide(input_images, 100000.0), 0., 1.), axis=-1)
    return input_data, output_label


print("Get Dataset Paths", end='')
start_time = time.time()
path = "dataset"
file_list = get_files(path)
file_list = np.array(file_list)
np.random.shuffle(file_list)
end_time = time.time()
print(f": Took {end_time - start_time} seconds")

train_files = file_list[:int(len(file_list)*0.8)]
val_files = file_list[int(len(file_list)*0.8):]

train_dataset = tf.data.Dataset.from_tensor_slices(train_files) \
    .shuffle(buffer_size=100000, seed=1213, reshuffle_each_iteration=True) \
    .batch(128*4) \
    .map(open_files_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .map(read_files_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .map(processing_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(val_files) \
    .batch(128*4) \
    .map(open_files_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .map(read_files_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .map(processing_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    topK = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)
    cnn = CNN()
    cnn.compile(optimizer=tf.keras.optimizers.Adam(jit_compile=False),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                         topK,
                         ])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)
cnn.fit(train_dataset,
        epochs=30,
        verbose=1,
        validation_data=val_dataset,
        use_multiprocessing=True,
        workers=20,
        max_queue_size=1,
        callbacks=[early_stopping]
        )


cnn.save(f'FinalCNN')
