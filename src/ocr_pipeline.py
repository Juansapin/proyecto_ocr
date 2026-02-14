import easyocr
from utils import preprocess_image


class OCRPipeline:
    def __init__(self, languages=["es", "en"]):
        """
        Inicializa el lector de EasyOCR.
        Para qué: Cargamos el modelo en memoria una sola vez para ahorrar tiempo.
        """
        self.reader = easyocr.Reader(languages, gpu=False)  # gpu=True si tienes NVIDIA

    def extract_text(self, image_path):
        """
        Procesa la imagen y extrae el texto.
        """
        # 1. Preprocesar
        processed_img = preprocess_image(image_path)

        # 2. Ejecutar OCR (detail=0 devuelve solo el texto limpio)
        results = self.reader.readtext(processed_img, detail=0)

        # 3. Unir resultados en un solo string
        return "\n".join(results)
