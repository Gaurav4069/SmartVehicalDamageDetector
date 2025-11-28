import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import json
import os

# === Paths ===
TRAIN_DIR = "../Stanford_Cars_dataset-main/dataset/train"
TEST_DIR = "../Stanford_Cars_dataset-main/dataset/test"

# === Parameters ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 25

# === Data Generators ===
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1]
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)

# === Load Base Model ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Phase 1: freeze base

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === Compile Model (Phase 1) ===
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1),
    ModelCheckpoint("model/best_model.keras", save_best_only=True)
]

# === Phase 1: Train top layers ===
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks
)

# === Phase 2: Fine-tune deeper layers ===
base_model.trainable = True
for layer in base_model.layers[:-50]:  # freeze all except last 50 layers
    layer.trainable = False

# Recompile with lower LR
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks
)

# === Combine History ===
history = {
    "accuracy": history1.history['accuracy'] + history2.history['accuracy'],
    "val_accuracy": history1.history['val_accuracy'] + history2.history['val_accuracy'],
    "loss": history1.history['loss'] + history2.history['loss'],
    "val_loss": history1.history['val_loss'] + history2.history['val_loss']
}

# === Save Final Model ===
os.makedirs("model", exist_ok=True)
model.save("model/car_model_detector.keras")

class_indices = {v: k for k, v in train_gen.class_indices.items()}
with open("model/car_labels.json", "w") as f:
    json.dump(class_indices, f)

print(" Model retrained and saved successfully!")

# === Evaluate on Test Data ===
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_gen)
print(f"🏁 Test Accuracy: {test_acc*100:.2f}%")

# === Plot Accuracy ===
plt.figure(figsize=(6, 4))
plt.plot(history["accuracy"], label='Train Accuracy', marker='o')
plt.plot(history["val_accuracy"], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model/accuracy_plot.png")
plt.show()

# === Plot Loss ===
plt.figure(figsize=(6, 4))
plt.plot(history["loss"], label='Train Loss', marker='o')
plt.plot(history["val_loss"], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model/loss_plot.png")
plt.show()
