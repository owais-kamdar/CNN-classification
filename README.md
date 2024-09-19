# ISIC Image Classification with CNN

## Project Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify medical images fetched from the **ISIC Archive API**. It includes the following steps:

1. Fetching and downloading images from the ISIC Archive.
2. Preprocessing the images and creating TensorFlow datasets.
3. Building and training a CNN model for image classification using TensorFlow/Keras.
4. Applying early stopping and model checkpointing to optimize the model training process.
5. Evaluating the trained model.

## Technologies Used
- **Python** (v3.10+)
- **TensorFlow**: Deep learning framework for building and training the CNN model.
- **Keras**: Used for building the model layers.
- **ISIC Archive API**: Used for fetching medical images.
- **Pillow (PIL)**: Image processing library for loading and resizing images.
- **Requests**: For making HTTP requests to the API.



## Installation and Setup

### 1. Clone the Repository
```bash
git clone <repository_url>
cd project-detection
```

### 2. Set up the Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```





### Project Workflow
Step 1: Fetch and Save Images
In api.py, the fetch_and_save_images function connects to the ISIC Archive API and downloads the medical images. You can specify the number of images you want to download.

Step 2: Preprocess the Images
The preprocess_image function in preprocess.py loads the images from the file system, resizes them to (224x224) pixels, and normalizes them. Then, create_tf_dataset generates a TensorFlow dataset ready for training.

Step 3: Build and Train the Model
The train_model function in model.py builds a CNN model using TensorFlow/Keras and applies early stopping and checkpointing to save the best model during training.

The model is then trained using the preprocessed image dataset with the following key features:

CNN Model: The core architecture includes convolutional layers, pooling, and dense layers for classification.

Batch Size: Images are processed in batches of size 16.

Steps per Epoch: We limit the number of steps per epoch to shorten training time.

Model Saving: The model is saved to final_model.keras during training.

Step 4: Model Evaluation
After training, you can evaluate the model performance by reviewing the training and validation accuracy, and loss values logged during the training process.

## Customization
You can modify the following aspects to fit your needs:

Number of Images: Adjust the number of images fetched from the API by modifying the limit parameter in fetch_and_save_images.

CNN Architecture: Change the model architecture in model.py to experiment with different deep learning architectures.

Training Parameters: You can modify parameters such as epochs, batch_size, steps_per_epoch, and validation_steps in train_model to optimize training speed and model performance.




## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.