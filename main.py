from src.api import fetch_and_save_images
from src.preprocess import create_tf_dataset
from src.model import train_model
import tensorflow as tf

# Step 1: Download images and fetch filenames and labels
image_filenames, label_list = fetch_and_save_images(limit=100)  # Fetch 100 images

# Ensure that we have matching labels and images
if len(image_filenames) != len(label_list):
    raise ValueError(f"Mismatch: {len(image_filenames)} images and {len(label_list)} labels.")

# Step 2: Split the data into training and validation datasets
train_size = int(0.8 * len(image_filenames))
train_filenames = image_filenames[:train_size]
train_labels = label_list[:train_size]

val_filenames = image_filenames[train_size:]
val_labels = label_list[train_size:]

# Print to ensure datasets are created
print(f"Train dataset size: {len(train_filenames)}")
print(f"Validation dataset size: {len(val_filenames)}")

# Step 3: Create the TensorFlow datasets with cache and reduced shuffle buffer size
batch_size = 16  # Reduced batch size
train_dataset = create_tf_dataset('images', train_filenames, train_labels, batch_size=batch_size)
train_dataset = train_dataset.shuffle(buffer_size=16)  # Reduce shuffle buffer
train_dataset = train_dataset.cache()  # Cache the dataset for faster processing
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for efficiency

val_dataset = create_tf_dataset('images', val_filenames, val_labels, batch_size=batch_size)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for efficiency

# Set steps_per_epoch and validation_steps to reduce the number of steps
steps_per_epoch = 10  # Try reducing this to speed up training
validation_steps = 5

# Step 4: Train the CNN model
model, history = train_model(
    train_dataset, 
    val_dataset, 
    train_size=len(train_filenames), 
    val_size=len(val_filenames), 
    steps_per_epoch=steps_per_epoch, 
    validation_steps=validation_steps
)

# Save the trained model and training history
model.save('final_model.keras')
print("Model training completed and saved.")
