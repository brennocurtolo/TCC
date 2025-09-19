import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]

import keras
from keras import layers
from keras import ops

import parameters
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# F1-Score via scikit-learn
from sklearn.metrics import f1_score

# Dynamic Learning Rate
from keras.callbacks import LearningRateScheduler
import math

# CutMix
import tensorflow_probability as tfp

def cutmix(images, labels, alpha=1.0):
    images = tf.cast(images, tf.float32) 
    batch_size = tf.shape(images)[0]

    # Shuffle dentro do batch
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # Lambda para interpolação
    lam = tf.random.uniform((batch_size,), 0.3, 0.7)

    # Bounding box
    image_dims = tf.shape(images)[1:3]
    cut_rat = tf.math.sqrt(1.0 - lam)
    cut_w = tf.cast(cut_rat * tf.cast(image_dims[1], tf.float32), tf.int32)
    cut_h = tf.cast(cut_rat * tf.cast(image_dims[0], tf.float32), tf.int32)

    # Centros
    cx = tf.random.uniform((batch_size,), 0, image_dims[1], dtype=tf.int32)
    cy = tf.random.uniform((batch_size,), 0, image_dims[0], dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_dims[1])
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_dims[0])
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_dims[1])
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_dims[0])

    def cutmix_single(image, shuffled_image, x1, y1, x2, y2):
        cropped_area = shuffled_image[y1:y2, x1:x2, :]
        paddings = [[y1, image_dims[0] - y2], [x1, image_dims[1] - x2], [0, 0]]
        patched = tf.pad(cropped_area, paddings, constant_values=0)
        mask = tf.pad(tf.ones_like(cropped_area), paddings, constant_values=0)
        return image * (1 - mask) + patched

    images = tf.map_fn(
        lambda elems: cutmix_single(*elems),
        (images, shuffled_images, x1, y1, x2, y2),
        dtype=tf.float32,
    )

    lam = tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(image_dims[0] * image_dims[1], tf.float32)
    labels = tf.one_hot(labels[:, 0], num_classes)
    shuffled_labels = tf.one_hot(shuffled_labels[:, 0], num_classes)
    mixed_labels = lam[:, None] * labels + (1 - lam[:, None]) * shuffled_labels

    return images, mixed_labels


# Parâmetros para warmup + cosine annealing
warmup_epochs = 5
initial_lr = 1e-6
target_lr = parameters.learning_rate
total_epochs = parameters.num_epochs

def lr_with_warmup_and_cosine(epoch):
    if epoch < warmup_epochs:
        return initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
    else:
        cosine_epoch = epoch - warmup_epochs
        cosine_total = total_epochs - warmup_epochs
        return target_lr * (1 + math.cos(math.pi * cosine_epoch / cosine_total)) / 2

num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(parameters.image_size, parameters.image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(parameters.patch_size)(augmented)
    encoded_patches = PatchEncoder(parameters.num_patches, parameters.projection_dim)(patches)

    for _ in range(parameters.transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=parameters.num_heads, key_dim=parameters.projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=parameters.transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=parameters.mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=target_lr, weight_decay=parameters.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/out/checkpoint-" + str(parameters.num_epochs) + "-cutmix.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    lr_scheduler = LearningRateScheduler(lr_with_warmup_and_cosine, verbose=1)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1024)
        .batch(parameters.batch_size)
        .map(lambda x, y: cutmix(x, y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    history = model.fit(
        train_ds,
        validation_data=(x_test, tf.one_hot(y_test[:, 0], num_classes)),
        epochs=parameters.num_epochs,
        callbacks=[checkpoint_callback, lr_scheduler],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, tf.one_hot(y_test[:, 0], num_classes))

    y_pred_logits = model.predict(x_test, batch_size=parameters.batch_size)
    y_pred = np.argmax(y_pred_logits, axis=1)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    print(f"Test F1-Score (macro): {round(f1_macro * 100, 2)}%")

    return history

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

def plot_learning_rate_schedule(total_epochs, initial_lr):
    epochs = np.arange(total_epochs)
    lrs = [lr_with_warmup_and_cosine(epoch) for epoch in epochs]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule (Warmup + Cosine Annealing)", fontsize=14)
    plt.grid()
    plt.show()

plot_history("loss")
plot_history("top-5-accuracy")
plot_learning_rate_schedule(parameters.num_epochs, parameters.learning_rate)
