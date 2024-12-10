import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(train_dir, val_dir, test_dir, img_height=224, img_width=224, batch_size=32):
    """
    Carga y preprocesa imágenes de rayos X para entrenamiento, validación y prueba.
    
    Args:
        train_dir (str): Directorio de imágenes de entrenamiento
        val_dir (str): Directorio de imágenes de validación
        test_dir (str): Directorio de imágenes de prueba
        img_height (int): Altura de redimensionamiento de imagen
        img_width (int): Ancho de redimensionamiento de imagen
        batch_size (int): Tamaño del lote
    
    Returns:
        tuple: Generadores de datos de entrenamiento, validación y prueba
    """
    # Configuración de aumentación de datos para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Configuración para validación y prueba (solo rescale)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generadores de datos
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, val_generator, test_generator

def print_class_distribution(generator):
    """
    Imprime la distribución de clases en el conjunto de datos.
    """
    class_indices = generator.class_indices
    for clase, indice in class_indices.items():
        print(f"Clase {clase}: {list(generator.classes).count(indice)} imágenes")