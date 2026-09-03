"""AgriVision AI - CNN training + real validation pipeline.

Final assessment mode expects the New Plant Diseases Dataset with:
  data/new_plant_diseases/train/<class>/*.jpg
  data/new_plant_diseases/valid/<class>/*.jpg

The train split is divided into training/validation. The separate valid split is
kept untouched and is used as the held-out TEST set for final metrics.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE / "data" / "new_plant_diseases"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "valid"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 20
SEED = 42

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise SystemExit(
        "Dataset not found. Put the Kaggle New Plant Diseases Dataset here:\n"
        "  data/new_plant_diseases/train/<class>/images\n"
        "  data/new_plant_diseases/valid/<class>/images"
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    validation_split=0.15, subset="training", seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    validation_split=0.15, subset="validation", seed=SEED
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False
)

class_names = train_ds.class_names
if class_names != test_ds.class_names:
    raise SystemExit("Training and test class folders do not match.")

autotune = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(autotune)
val_ds = val_ds.prefetch(autotune)
test_ds = test_ds.prefetch(autotune)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.35),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

out = BASE / "metrics"
out.mkdir(exist_ok=True)
(BASE / "models").mkdir(exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(BASE / "models" / "best_crop_disease_cnn.keras",
                                       monitor="val_accuracy", save_best_only=True)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
model.save(BASE / "models" / "crop_disease_cnn.keras")

# Held-out evaluation: test_ds is never used for fitting.
y_true, y_pred = [], []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(probs, axis=1).tolist())

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

# Save a small auditable set of actual-vs-predicted examples for the website.
test_files = []
valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
for cls in class_names:
    folder = TEST_DIR / cls
    test_files.extend(sorted([str(x.relative_to(TEST_DIR)).replace("\\", "/") for x in folder.rglob("*") if x.suffix.lower() in valid_ext]))
samples = []
for idx in range(min(24, len(y_true), len(test_files))):
    actual = class_names[int(y_true[idx])]
    predicted = class_names[int(y_pred[idx])]
    samples.append({"image": test_files[idx], "actual": actual, "predicted": predicted,
                    "correct": bool(y_true[idx] == y_pred[idx])})

report = classification_report(y_true, y_pred, target_names=class_names,
                               output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred).tolist()

metrics = {
    "dataset": "New Plant Diseases Dataset",
    "classes": len(class_names),
    "test_images": int(len(y_true)),
    "accuracy": float(report["accuracy"]),
    "precision_macro": float(report["macro avg"]["precision"]),
    "recall_macro": float(report["macro avg"]["recall"]),
    "f1_macro": float(report["macro avg"]["f1-score"]),
    "classification_report": report,
    "confusion_matrix": cm,
    "class_names": class_names,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
(out / "validation_samples.json").write_text(json.dumps(samples, indent=2))
(out / "classes.json").write_text(json.dumps(class_names, indent=2))
(out / "training_history.json").write_text(json.dumps(history.history, indent=2))

print("\n=== HELD-OUT TEST RESULTS ===")
print(f"Test images : {len(y_true)}")
print(f"Accuracy    : {report['accuracy']*100:.2f}%")
print(f"Precision   : {report['macro avg']['precision']*100:.2f}%")
print(f"Recall      : {report['macro avg']['recall']*100:.2f}%")
print(f"F1 Score    : {report['macro avg']['f1-score']*100:.2f}%")
print(f"Metrics     : {out / 'metrics.json'}")
print(f"Model       : {BASE / 'models' / 'crop_disease_cnn.keras'}")
