import unittest
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


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_image(self):
        image_path = "images/ISIC_0485014.jpg"  # Make sure this path exists for testing
        processed_img = preprocess_image(image_path)
        self.assertEqual(processed_img.shape, (224, 224, 3))

if __name__ == "__main__":
    unittest.main()
