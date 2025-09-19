import os
# os.environ["KERAS_BACKEND"] = "tensorflow"  # Removido porque não é necessário

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler

import parameters

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

# Remove a dimensão extra e aplica one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train.squeeze(), num_classes)
y_test = tf.keras.utils.to_categorical(y_test.squeeze(), num_classes)


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
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(
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
        positions = tf.expand_dims(
            tf.range(start=0, limit=self.num_patches, delta=1), axis=0
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
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/out/checkpoint-" + str(parameters.num_epochs) + ".weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    lr_scheduler = LearningRateScheduler(lr_with_warmup_and_cosine, verbose=1)

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=parameters.batch_size,
        epochs=parameters.num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback, lr_scheduler],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)

    y_pred_logits = model.predict(x_test, batch_size=parameters.batch_size)
    y_pred = np.argmax(y_pred_logits, axis=1)

    y_true = np.argmax(y_test, axis=1)

    f1_macro = f1_score(y_true, y_pred, average='macro')


    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    print(f"Test F1-Score (macro): {round(f1_macro * 100, 2)}%")

    return history

def check_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=parameters.learning_rate, weight_decay=parameters.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoints = ["checkpoint-20.weights.h5"]

    for checkpoint_filepath in checkpoints:
        model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)

        y_pred_logits = model.predict(x_test, batch_size=parameters.batch_size)
        y_pred = np.argmax(y_pred_logits, axis=1)

        # Converte y_test one-hot de volta para inteiros
        y_true = np.argmax(y_test, axis=1)

        f1_macro = f1_score(y_true, y_pred, average='macro')


        print()
        print('File Name : ', checkpoint_filepath)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
        print(f"Test F1-Score (macro): {round(f1_macro * 100, 2)}%")
        print()

    return model

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
