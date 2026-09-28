"""
train_model.py
----------------
Trains a plant disease classifier using transfer learning (MobileNetV2).

Expected dataset layout (e.g. the PlantVillage dataset from Kaggle:
https://www.kaggle.com/datasets/emmarex/plantdisease):

    dataset/
        Tomato___Late_blight/
            img1.jpg
            img2.jpg
            ...
        Tomato___healthy/
            img1.jpg
            ...
        Potato___Early_blight/
            ...
        ... (one folder per class)

Usage:
    python train_model.py --data_dir ./dataset --epochs 15 --batch_size 32
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)


def build_model(num_classes: int) -> tf.keras.Model:
    """Builds a MobileNetV2-based transfer learning model."""
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze base for initial training

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def get_generators(data_dir: str, batch_size: int):
    """Creates train/validation generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0,  # preprocessing handled inside the model
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


def plot_history(history, out_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to dataset folder (one subfolder per class)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_gen, val_gen = get_generators(args.data_dir, args.batch_size)
    num_classes = train_gen.num_classes
    print(f"Found {num_classes} classes: {train_gen.class_indices}")

    # Save class index mapping (needed at inference time)
    class_indices = {v: k for k, v in train_gen.class_indices.items()}
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_indices, f, indent=2)

    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
    ]

    print("\n=== Stage 1: training classifier head (base frozen) ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\n=== Stage 2: fine-tuning top layers of base model ===")
    base_model.trainable = True
    # Freeze all but the last ~30 layers for fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    final_path = os.path.join(args.output_dir, "plant_disease_model.keras")
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")

    # Combine histories for plotting
    for k in history.history:
        history.history[k] += history_fine.history[k]
    plot_history(history, os.path.join(args.output_dir, "training_history.png"))

    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
