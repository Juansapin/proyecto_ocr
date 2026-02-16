### PIPE LINE DEL PRE PROCESAMIENTO ###

# Importación de las librerías

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Se crea una clase que contiene todas las funciones a utlizar para el preprocesamiento lo que permite la clase es poder escoger o utilizar diferentes configuraciones
# de parametros de los diferentes pasos del preprocesamiento para poder utilizarlos a la hora de hacer el tratamiento de las imagenes
class OCRPrepocesador:
    def __init__(self, tamano_estandar: int = 128):
        """
        tamano_estandar: Tamaño estandar para las imagenes
        """
        self.tamano_estandar = tamano_estandar
        self.pasos_preprocesamiento = []
        
    def cargar_imagen(self, ruta_imagen: str) -> np.ndarray:
        """Carga una imagen desde la ruta"""
        img = cv2.imread(ruta_imagen)
        if img is None:
            raise ValueError(f"No se encuentra la ruta: {ruta_imagen}")
        return img
    
    def from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """Converitr a formato OpenCV."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # PASO 1: Convetir a escala de grises cuando la imagen viene a color
    
    def escala_grises(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3: # 3 es porque es RGB 
            gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gris = image.copy()
        
        self.pasos_preprocesamiento.append("Escala de grises")
        return gris
    
    # PASO 2: Reducción de ruido, elimina el ruido que puede traer la imagen
    
    def quita_ruido(self, image: np.ndarray, metodo: str = 'gaussian') -> np.ndarray:
        """
        Metodos:
        - gaussian: Bueno para ruido gaussiano, produce resultados suaves
        - median: Excelente para ruido sal y pimienta, preserva bordes
        - bilateral: Preserva bordes mientras suaviza, ideal para documentos con detalle
        - nlm: Non-Local Means, mejor calidad pero más lento
        """

        if metodo == 'gaussian':
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        elif metodo == 'median':
            denoised = cv2.medianBlur(image, 5)
        elif metodo == 'bilateral':
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        elif metodo == 'nlm':
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            raise ValueError(f"Metodo desconocido: {metodo}")
        
        self.pasos_preprocesamiento.append(f"Sin ruido metodo: ({metodo})")
        return denoised
    
    # PASO 3: Binarización: Convierte la imagen a blanco y negro para hacer el contraste entre las palabras y el fondo
    
    def binarize(self, image: np.ndarray, metodo: str = 'otsu') -> np.ndarray:
        """
        Métodos:
        - otsu: Selección automática del umbral, funciona bien con iluminación uniforme
        - adaptive_mean: Bueno para iluminación variable (sombras, gradientes)
        - adaptive_gaussian: Ideal para condiciones de iluminación complejas
        - sauvola: Excelente para documentos históricos con degradación
        """

        if metodo == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif metodo == 'adaptive_mean':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif metodo == 'adaptive_gaussian':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif metodo == 'sauvola':
            # Sirve para documentos degradados
            window_size = 25
            k = 0.2
            R = 128
            
            mean = cv2.blur(image.astype(np.float64), (window_size, window_size))
            mean_sq = cv2.blur(image.astype(np.float64)**2, (window_size, window_size))
            std = np.sqrt(mean_sq - mean**2)
            
            # Sauvola threshold
            threshold = mean * (1 + k * ((std / R) - 1))
            binary = np.where(image > threshold, 255, 0).astype(np.uint8)
            
        else:
            raise ValueError(f"Metodo de binarización desconocido: {metodo}")
        
        self.pasos_preprocesamiento.append(f"Binarizacion metodo: ({metodo})")
        return binary
    
    # PASO 4: Operaciones morfologicas, estas operaciones sirven para limpiar la imagen en blanco y negro y elimiar ruido resultante de la transformación anterior
    
    def operaciones_morfologicas(self, image: np.ndarray, 
                                 operacion: str = 'opening',
                                 kernel_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        Operaciones:
        - opening: Elimina pequeño ruido blanco (erosión seguida de dilatación)
        - closing: Rellena pequeños huecos en los caracteres (dilatación seguida de erosión)
        - dilation: Engrosa los caracteres (útil para texto delgado o desvanecido)
        - erosion: Adelgaza los caracteres (útil para texto grueso o en negrita)

        """
        # Se crea un kernel de 2x2 para eliminar el ruido dentro de cada imagen convertida en blanco y negro, es de 2x2 porque así no cambia mucho el texto 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        if operacion == 'opening':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operacion == 'closing':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operacion == 'dilation':
            result = cv2.dilate(image, kernel, iterations=1)
        elif operacion == 'erosion':
            result = cv2.erode(image, kernel, iterations=1)
        else:
            raise ValueError(f"Unknown operation: {operacion}")
        
        self.pasos_preprocesamiento.append(f"Morfologia metodo: ({operacion})")
        return result
    
    # PASO 5: Inclinacion, el modelo funciona mejor si el texto en las imagene esta recto, por lo que se ajustan posibles inclinaciones
    
    def proceso_rotacion(self, image: np.ndarray) -> np.ndarray:

        # Identifirca los bordes
        bordes = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Saca las lineas usando Hough
        lineas = cv2.HoughLines(bordes, 1, np.pi/180, 100)
        
        if lineas is not None and len(lineas) > 0:
            # Calcula el angulo promedio
            angulos = []
            for rho, theta in lineas[:, 0]:
                angulo = np.degrees(theta) - 90
                angulos.append(angulo)
            
            angulo_promedio = np.median(angulos)
            
            # Solo ajusta si la inclinación es alta (> 0.5)
            if abs(angulo_promedio) > 0.5:
                # Calcula el centro de la magen
                (h, w) = image.shape[:2]
                centro = (w // 2, h // 2)
                
                # Gira la imagen
                M = cv2.getRotationMatrix2D(centro, angulo_promedio, 1.0)
                rotada = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                self.pasos_preprocesamiento.append(f"Se elimina la rotacion ({angulo_promedio:.2f}°)")
                return rotada
        
        self.pasos_preprocesamiento.append("Sin rotacion")
        return image
    
    # PASO 6: Eliminar bordes
    
    def quitar_bordes(self, image: np.ndarray, tamano_borde: int = 10) -> np.ndarray:
        h, w = image.shape[:2]
        cortado = image[tamano_borde:h-tamano_borde, tamano_borde:w-tamano_borde]
        
        self.pasos_preprocesamiento.append(f"Se quita el borde ({tamano_borde}px)")
        return cortado
    
    # PASO 7: Mejora contraste, los modelos necesitan que el fondo se distinga del texto
    
    def mejora_contraste(self, image: np.ndarray, metodo: str = 'clahe') -> np.ndarray:
        """
        Métodos:
        - clahe: Ecualización adaptativa del histograma, ideal para contraste variable
        - histogram: Ecualización global del histograma, adecuada para bajo contraste uniforme
        - gamma: Corrección gamma, útil para ajustar el brillo general
        """

        if metodo == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrastada = clahe.apply(image)
            
        elif metodo == 'histogram':
            contrastada = cv2.equalizeHist(image)
            
        elif metodo == 'gamma':
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 
                for i in range(256)
            ]).astype(np.uint8)
            contrastada = cv2.LUT(image, table)
            
        else:
            raise ValueError(f"Unknown enhancement metodo: {metodo}")
        
        self.pasos_preprocesamiento.append(f"Contrast Enhancement ({metodo})")
        return contrastada
    
    # PASO 8: Resize, re ajustar el tamaño de las imagenes para que puedan ser interpretadas por el modelo
    
    def resize_image(self, image: np.ndarray, 
                    height: Optional[int] = None) -> np.ndarray:
        if height is None:
            height = self.target_height
        
        h, w = image.shape[:2]
        aspect_ratio = w / h
        new_width = int(height * aspect_ratio)
        
        resized = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)
        
        self.pasos_preprocesamiento.append(f"Resize (h={height})")
        return resized
    
    # Pipeline completo, esta función permite ejecutar todos los pasos dependiendo de los parametros de las subfunciones
    def preprocess(self, image: np.ndarray, 
                   pipeline: str = 'standard') -> np.ndarray:
        """
        Apply a complete preprocessing pipeline.
        
        Pipelines:
        - 'standard': Basic pipeline for clean scans
        - 'aggressive': For poor quality or degraded documents
        - 'minimal': Quick preprocessing for high-quality images
        - 'handwriting': Optimized for handwritten text
        
        Args:
            image: Input image (BGR or grayscale)
            pipeline: Pipeline type
            
        Returns:
            Preprocessed image
        """
        self.pasos_preprocesamiento = []
        
        if pipeline == 'standard':
            # Standard pipeline for typical scanned documents
            img = self.escala_grises(image)
            img = self.quita_ruido(img, metodo='gaussian')
            img = self.mejora_contraste(img, metodo='clahe')
            img = self.binarize(img, metodo='adaptive_gaussian')
            img = self.operaciones_morfologicas(img, operation='opening')
            img = self.proceso_rotacion(img)
            img = self.resize_image(img)
            
        elif pipeline == 'aggressive':
            # Aggressive pipeline for degraded documents
            img = self.escala_grises(image)
            img = self.quita_ruido(img, metodo='nlm')
            img = self.mejora_contraste(img, metodo='clahe')
            img = self.binarize(img, metodo='sauvola')
            img = self.operaciones_morfologicas(img, operation='closing')
            img = self.operaciones_morfologicas(img, operation='opening')
            img = self.proceso_rotacion(img)
            img = self.quitar_bordes(img, border_size=5)
            img = self.resize_image(img)
            
        elif pipeline == 'minimal':
            # Minimal pipeline for high-quality images
            img = self.escala_grises(image)
            img = self.binarize(img, metodo='otsu')
            img = self.resize_image(img)
            
        elif pipeline == 'handwriting':
            # Optimized for handwritten text (like IAM dataset)
            img = self.escala_grises(image)
            img = self.quita_ruido(img, metodo='bilateral')
            img = self.mejora_contraste(img, metodo='clahe')
            img = self.binarize(img, metodo='adaptive_gaussian')
            img = self.operaciones_morfologicas(img, operation='opening', kernel_size=(2, 2))
            img = self.resize_image(img)
            
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        
        return img
    
    def reumen_pipeline(self) -> str:
        return " → ".join(self.pasos_preprocesamiento)



if __name__ == "__main__":
    print("=== DEMO INTERACTIVA ===\n")
    
    # Cargar imagen de ejemplo
    import cv2
    sample = cv2.imread('sample.jpg')
    
    # Crear preprocessor
    prep = OCRPrepocesador()
    
    # Mostrar resultados
    result = prep.preprocess(sample, pipeline='standard')
    cv2.imshow('Resultado', result)
    cv2.waitKey(0)
    
    print("Demo completada!")

