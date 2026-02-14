import cv2
import numpy as np


def preprocess_image(image_path):
    """
    Lee una imagen y aplica filtros para mejorar la detección de texto.
    Por qué: El contraste ayuda a que el modelo no confunda manchas con letras.
    """
    # Cargar imagen
    img = cv2.imread(image_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbralado para limpiar ruido (Opcional pero recomendado para libros)
    # _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray
