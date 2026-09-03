import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "data" / "train" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test" / "test"

class Model:
    def __init__(self, n_classes: int, input_shape: tuple) -> None:
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.model = None

    @staticmethod
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (32, 32))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def create_train_val_dataset(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create train and validation datasets from the labeled training CSV."""
        df = pd.read_csv(PROJECT_ROOT / "data" / "trainLabels.csv")

        df["id"] = df["id"].astype(str)
        df["image_path"] = df["id"].map(lambda x: str(TRAIN_DIR / f"{x}.png"))

        label_to_idx = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
        df["label_idx"] = df["label"].map(label_to_idx).astype(np.int32)

        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["label_idx"],
        )

        X_train = train_df["image_path"].to_numpy()
        y_train = train_df["label_idx"].to_numpy().astype(np.int32)

        X_val = val_df["image_path"].to_numpy()
        y_val = val_df["label_idx"].to_numpy().astype(np.int32)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(32).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    def train_model(self):
        train_ds, val_ds = self.create_train_val_dataset()

        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(self.n_classes, activation="softmax"),
        ])

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        print("--- Initiating training sequence ---")
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=15,
        )
        print("--- Finished ---")
        return history

    def display_model_summary(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.summary()

    def save_model(self, name: str = "image_classification_model"):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.save(f"{PROJECT_ROOT}/src/image_recognition_network/models/{name}.h5")


if __name__ == "__main__":
    model = Model(10, (32, 32, 3))
    model.train_model()
    model.display_model_summary()
    model.save_model("image_classification_model")