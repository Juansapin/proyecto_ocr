import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Prepara la imagen para mejorar la precisión del OCR.
    """
    # Leer imagen
    img = cv2.imread(image_path)
    
    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Eliminar ruido (Denoising)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 3. Binarización adaptativa (útil para iluminación no uniforme en libros)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh