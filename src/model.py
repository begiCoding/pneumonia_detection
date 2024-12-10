
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Crea un modelo CNN para clasificación de neumonía.
    
    Args:
        input_shape (tuple): Dimensiones de entrada de la imagen
        num_classes (int): Número de clases de salida
    
    Returns:
        tensorflow.keras.Model: Modelo CNN compilado
    """
    model = Sequential([
        # Bloque convolucional 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Bloque convolucional 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Bloque convolucional 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Capas densas
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model