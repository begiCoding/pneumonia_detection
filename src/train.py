import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
from model import create_cnn_model
from preprocess import load_and_preprocess_data, print_class_distribution

def train_model(train_dir, val_dir, test_dir, epochs=20, batch_size=32):
    """
    Entrena un modelo de CNN para detección de neumonía.
    
    Args:
        train_dir (str): Directorio de imágenes de entrenamiento
        val_dir (str): Directorio de imágenes de validación
        test_dir (str): Directorio de imágenes de prueba
        epochs (int): Número de épocas de entrenamiento
        batch_size (int): Tamaño del lote
    """
    # Crear directorio de resultados si no existe
    os.makedirs('results', exist_ok=True)
    
    # Cargar y preprocesar datos
    train_generator, val_generator, test_generator = load_and_preprocess_data(
        train_dir, val_dir, test_dir, batch_size=batch_size
    )
    
    # Imprimir distribución de clases
    print("Distribución de clases en entrenamiento:")
    print_class_distribution(train_generator)
    
    # Crear modelo
    model = create_cnn_model()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'results/best_model.keras', 
        monitor='val_accuracy', 
        save_best_only=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    # Entrenar modelo
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Evaluar modelo
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test AUC: {test_auc}")
    
    # Guardar historia de entrenamiento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

if __name__ == "__main__":
    train_model(
        train_dir='data/train', 
        val_dir='data/val', 
        test_dir='data/test'
    )