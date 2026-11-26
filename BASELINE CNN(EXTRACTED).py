
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET_PATH = Path("bioprinting_sample")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

CLASSES = ["good", "under", "over"]
CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASSES)}

# ============================================================
# 3. COLLECT IMAGES
# ============================================================

categories = [
    ("real_data", "augmented_real_infill"),
    ("real_data", "augmented_real_line"),
    ("synth_data", "augmented_synth_infill"),
    ("synth_data", "augmented_synth_line")
]

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

train_paths = []
train_labels = []

test_paths = []
test_labels = []

for parent, category in categories:

    for split in ["train", "test"]:

        for quality in CLASSES:

            folder = (
                DATASET_PATH /
                parent /
                category /
                split /
                quality
            )

            if not folder.exists():
                raise FileNotFoundError(f"Missing folder: {folder}")

            images = sorted([
                str(p) for p in folder.iterdir()
                if p.suffix.lower() in extensions
            ])

            label = CLASS_TO_LABEL[quality]

            if split == "train":
                train_paths.extend(images)
                train_labels.extend([label] * len(images))
            else:
                test_paths.extend(images)
                test_labels.extend([label] * len(images))

train_paths = np.array(train_paths)
train_labels = np.array(train_labels)

test_paths = np.array(test_paths)
test_labels = np.array(test_labels)

print("Total training images:", len(train_paths))
print("Total testing images:", len(test_paths))

# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    train_paths,
    train_labels,
    test_size=0.20,
    random_state=SEED,
    stratify=train_labels
)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(test_paths))

# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    return dataset.prefetch(tf.data.AUTOTUNE)


train_ds = create_dataset(X_train, y_train, training=True)
val_ds = create_dataset(X_val, y_val)
test_ds = create_dataset(test_paths, test_labels)

# ============================================================
# 6. BUILD BASELINE CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128, 128, 3)),

    # Convolution Block 1
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution Block 2
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution Block 3
    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Classification Head
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 7. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_baseline_cnn.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# ============================================================
# 8. TRAIN MODEL
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model
model.save("baseline_cnn_final.keras")

# ============================================================
# 9. PLOT TRAINING ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    marker="o"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    marker="o"
)

plt.title("Baseline CNN - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("baseline_accuracy.png", dpi=300)
plt.show()

# ============================================================
# 10. PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    marker="o"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    marker="o"
)

plt.title("Baseline CNN - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("baseline_loss.png", dpi=300)
plt.show()

# ============================================================
# 11. EVALUATE ON TEST DATA
# ============================================================

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTEST RESULTS")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))

# ============================================================
# 12. PREDICTIONS
# ============================================================

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    y_pred,
    labels=[0, 1, 2]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

plt.title("Baseline CNN - Confusion Matrix")
plt.tight_layout()
plt.savefig("baseline_confusion_matrix.png", dpi=300)
plt.show()

# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        test_labels,
        y_pred,
        target_names=CLASSES,
        digits=4
    )
)

# ============================================================
# 15. SAVE EXPERIMENT RESULTS
# ============================================================

os.makedirs("baseline_results", exist_ok=True)

import pandas as pd

results = pd.DataFrame({
    "Actual": [CLASSES[i] for i in test_labels],
    "Predicted": [CLASSES[i] for i in y_pred]
})

results.to_csv(
    "baseline_results/predictions.csv",
    index=False
)

print("\nBaseline CNN training completed!")
