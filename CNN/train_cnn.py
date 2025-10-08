import os
import random
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from face_detector import detect_faces  # import your DNN face detector

# ----------------- CONFIG -----------------
DATA_DIR = r"C:\Users\PC\Desktop\Smart-Attendance-System\data"  # your dataset folder
LABELS_CSV = os.path.join(DATA_DIR, "details.csv")               # label CSV
MODEL_DIR = "models"
MODEL_NAME = "smart_attendance_effb0"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
SEED = 42
FINETUNE_AT = 350
THRESHOLD = 0.80

os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------- HELPERS -----------------
def parse_person_folder(name: str):
    """
    Expect folder like: '1234_G1_John_Doe'
    Returns dict: {system_id, group, person_name}
    """
    parts = name.split("_")
    if len(parts) >= 3 and parts[1].upper().startswith("G"):
        system_id = parts[0]
        group = parts[1]
        person_name = "_".join(parts[2:])
    else:
        system_id = "Unknown"
        group = "Unknown"
        person_name = name
    return dict(system_id=system_id, group=group, person_name=person_name)

def build_label_table(data_dir: str, existing_csv: str):
    """
    Build a DataFrame mapping class_index -> folder_name + parsed metadata.
    Merge with existing details.csv if available.
    """
    folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    df = pd.DataFrame({"class_index": list(range(len(folders))), "folder_name": folders})
    meta = [parse_person_folder(f) for f in folders]
    meta_df = pd.DataFrame(meta)
    df = pd.concat([df, meta_df], axis=1)

    if os.path.exists(existing_csv):
        try:
            lab = pd.read_csv(existing_csv)
            if "Person_Name" in lab.columns:
                df = df.merge(lab, how="left", left_on="person_name", right_on="Person_Name")
        except Exception:
            pass

    return df

def make_datasets(data_dir, img_size, batch_size, val_split, test_split, seed):
    """
    Create train, validation, and test datasets using Keras image_dataset_from_directory.
    """
    ds_all = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split + test_split,
        subset="training"
    )
    ds_valtest = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split + test_split,
        subset="validation"
    )
    val_batches = int(len(ds_valtest) * (val_split / (val_split + test_split)))
    ds_val = ds_valtest.take(val_batches)
    ds_test = ds_valtest.skip(val_batches)
    return ds_all, ds_val, ds_test

def aug_block():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.12),
        layers.RandomTranslation(0.04, 0.04),
    ], name="augment")

def build_model(num_classes: int, img_size):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = aug_block()(x)

    base = keras.applications.EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="smart_attendance_cnn")
    return model, base

def set_finetune_layers(base_model, finetune_at: int):
    for layer in base_model.layers[-finetune_at:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

# ----------------- MAIN -----------------
if __name__ == "__main__":
    label_table = build_label_table(DATA_DIR, LABELS_CSV)
    label_table.to_csv(os.path.join(MODEL_DIR, f"{MODEL_NAME}_label_table.csv"), index=False)

    ds_train, ds_val, ds_test = make_datasets(DATA_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, SEED)

    AUTOTUNE = tf.data.AUTOTUNE
    ds_train = ds_train.prefetch(AUTOTUNE)
    ds_val = ds_val.cache().prefetch(AUTOTUNE)
    ds_test = ds_test.cache().prefetch(AUTOTUNE)

    all_labels = np.concatenate([y.numpy() for _, y in ds_train.unbatch().batch(1024)])
    classes = np.unique(all_labels)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=all_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    num_classes = len(classes)
    print(f"Classes: {num_classes}")
    print("Class weights:", class_weight_dict)

    model, base = build_model(num_classes, IMG_SIZE)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    ckpt_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-6, monitor="val_loss")
    ]

    # Phase 1: train classifier head
    history = model.fit(
        ds_train,
        epochs=EPOCHS//2,
        validation_data=ds_val,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Phase 2: fine-tune last layers
    set_finetune_layers(base, FINETUNE_AT)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history_ft = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Evaluate
    print("\nEvaluating on test set…")
    model.load_weights(ckpt_path)
    test_images, test_labels = [], []
    for x, y in ds_test.unbatch().batch(256):
        test_images.append(x.numpy())
        test_labels.append(y.numpy())
    Xte = np.concatenate(test_images)
    yte = np.concatenate(test_labels)
    preds = model.predict(Xte, batch_size=64)
    yhat = preds.argmax(axis=1)

    print(classification_report(yte, yhat, digits=4))
    cm = confusion_matrix(yte, yhat)
    np.savetxt(os.path.join(MODEL_DIR, f"{MODEL_NAME}_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    # Save config for inference
    config = {
        "img_size": IMG_SIZE,
        "threshold": THRESHOLD,
        "class_indices": {int(k): v for k, v in enumerate(sorted(os.listdir(DATA_DIR)))}
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_NAME}_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved best model to: {ckpt_path}")
    print(f"Label table: {os.path.join(MODEL_DIR, f'{MODEL_NAME}_label_table.csv')}")
    print("Done.")
