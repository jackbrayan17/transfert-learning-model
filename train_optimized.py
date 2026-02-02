import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Configuration
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
DATA_DIR = 'data/riceleaf'
EPOCHS_WARMUP = 10
EPOCHS_FINE = 20

print(f"Using TensorFlow Version: {tf.__version__}")

# 1. Data Loading
print("Loading Data...")
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'val'),
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# Optimize Data Loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Data Augmentation
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomContrast(0.2),
  layers.RandomBrightness(0.2),
])

# 3. Model Construction (EfficientNetB0)
print("Building Model...")
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

base_model = applications.EfficientNetB0(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze for warmup

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x) # Added dense layer for better feature mapping
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_FineTuned")

# 4. Stage 1: Warmup Training
print("\n=== Stage 1: Warmup (Frozen Base) ===")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_warmup = model.fit(train_ds, 
                           validation_data=val_ds, 
                           epochs=EPOCHS_WARMUP)

# 5. Stage 2: Fine-Tuning
print("\n=== Stage 2: Fine-Tuning (Unfrozen Top Layers) ===")
# Unfreeze the base model
base_model.trainable = True

# Tune only the top layers of the base model (e.g., last 30)
# This prevents destroying weights learned from ImageNet too quickly
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with low learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    callbacks.ModelCheckpoint('best_tuned_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
]

history_fine = model.fit(train_ds, 
                         validation_data=val_ds, 
                         epochs=EPOCHS_FINE,
                         callbacks=callbacks_list)

print("Training Complete.")

# 6. Save Metadata
print("Saving metadata...")
with open('best_tuned_model_info.json', 'w') as f:
    best_val_acc = max(history_fine.history['val_accuracy'])
    json.dump({
        "model_name": "EfficientNetB0_FineTuned", 
        "accuracy": best_val_acc,
        "class_names": class_names
    }, f)

# 7. Plotting Results
def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([EPOCHS_WARMUP-1, EPOCHS_WARMUP-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([EPOCHS_WARMUP-1, EPOCHS_WARMUP-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('optimization_results.png')
    print("Saved optimization_results.png")

plot_history(history_warmup, history_fine)

# Fix quantization config immediately just in case
import zipfile
import shutil

def fix_quantization(model_path):
    print(f"Applying quantization fix to {model_path}...")
    extract_dir = 'temp_fix_extract'
    if os.path.exists(extract_dir): shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)
    
    try:
        with zipfile.ZipFile(model_path, 'r') as z_in:
            z_in.extractall(extract_dir)
        
        config_path = os.path.join(extract_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: config = json.load(f)
            
            def clean(obj):
                if isinstance(obj, dict):
                    if 'quantization_config' in obj: obj.pop('quantization_config')
                    for k, v in obj.items(): clean(v)
                elif isinstance(obj, list):
                    for i in obj: clean(i)
            
            clean(config)
            with open(config_path, 'w') as f: json.dump(config, f)
            
            temp_name = model_path + ".temp"
            with zipfile.ZipFile(temp_name, 'w', zipfile.ZIP_DEFLATED) as z_out:
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        p = os.path.join(root, file)
                        z_out.write(p, os.path.relpath(p, extract_dir))
            shutil.move(temp_name, model_path)
            print("Fix applied successfully.")
    except Exception as e:
        print(f"Fix failed: {e}")
    finally:
        if os.path.exists(extract_dir): shutil.rmtree(extract_dir)

fix_quantization('best_tuned_model.keras')
