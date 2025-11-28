# ✅ IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import MobileNetV2
from keras.optimizers import Adam

# ✅ DATA PATHS
train_dir = '../data3a/training'
val_dir = '../data3a/validation'

# ✅ DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

num_classes = train_generator.num_classes
print(f"\n✅ Classes detected: {train_generator.class_indices}")

# ✅ BASE MODEL (Transfer Learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze base

# ✅ CUSTOM CLASSIFICATION HEAD
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ✅ COMPILE MODEL
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ CALLBACKS
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_vehicle_damage_model.keras', monitor='val_accuracy', save_best_only=True)

# ✅ TRAIN (Transfer Learning Stage)
print("\n🚀 Training Transfer Learning Model...")
history_1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# ✅ UNFREEZE LAST 30 LAYERS FOR FINE-TUNING
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# ✅ RE-COMPILE FOR FINE-TUNING
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ FINE-TUNE
print("\n🔧 Fine-tuning last layers...")
history_2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop]
)

# ✅ SAVE FINAL MODEL
model.save("final_vehicle_damage_model.keras")
print("\n✅ Model saved as 'final_vehicle_damage_model.keras'")

# ✅ MERGE HISTORIES FOR PLOTTING
acc = history_1.history['accuracy'] + history_2.history['accuracy']
val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
loss = history_1.history['loss'] + history_2.history['loss']
val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

# ✅ PLOT METRICS
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
