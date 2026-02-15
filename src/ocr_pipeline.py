import easyocr
import numpy as np

class BookOCR:
    def __init__(self, lang='es'):
        # Inicializa el lector para español (y otros idiomas si quieres, ej: ['es', 'en'])
        # gpu=False asegura que use tu procesador AMD sin buscar CUDA de Nvidia
        self.reader = easyocr.Reader([lang], gpu=False)

    def extract_text(self, processed_img):
        # processed_img puede ser la ruta o el array de OpenCV
        # detail=0 hace que solo devuelva el texto, sin coordenadas
        result = self.reader.readtext(processed_img, detail=0)
        
        if not result:
            return "No se detectó texto en la imagen."
            
        return "\n".join(result)