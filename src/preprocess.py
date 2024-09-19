# src/preprocess.py
import os
import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image_path):
    """Loads and preprocesses a single image from the file system."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to (224, 224) as required by the model
    img = np.array(img) / 255.0   # Normalize to [0, 1]
    return img

def create_tf_dataset(image_dir, image_filenames, label_list, batch_size=32):
    """Creates a TensorFlow dataset from preprocessed images and their corresponding labels."""
    
    def load_image(image_path):
        img = preprocess_image(image_path)
        return img

    def gen():
        for img_name, labels in zip(image_filenames, label_list):
            image_path = os.path.join(image_dir, f"{img_name}.jpg")
            img = load_image(image_path)
            yield img, labels
    
    # Define output signature for the dataset
    output_signature = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # Image tensor
        tf.TensorSpec(shape=(None,), dtype=tf.int32)           # Labels (variable-length array)
    )
    
    # Create the dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )
    
    # Shuffle, repeat, and batch the dataset
    dataset = dataset.shuffle(buffer_size=100).repeat().batch(batch_size)
    return dataset

