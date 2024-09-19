# src/model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_cnn_model():
    """Build and compile a CNN model for binary classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dataset, val_dataset, train_size, val_size, batch_size=32, steps_per_epoch=None, validation_steps=None):
    """Train the CNN model and apply early stopping and checkpointing."""
    
    # Build the CNN model (you should define the function build_cnn_model)
    model = build_cnn_model()

    # Callbacks for early stopping and model checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    # Calculate steps per epoch if not passed as arguments
    if steps_per_epoch is None:
        steps_per_epoch = train_size // batch_size
    if validation_steps is None:
        validation_steps = val_size // batch_size

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,  # You can adjust this as needed
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[early_stopping, checkpoint]
    )
    
    return model, history


