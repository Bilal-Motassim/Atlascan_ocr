from paddleocr import PaddleOCR
from utils.llama_utils import generate_structured_data_with_llama2

ocr = PaddleOCR(use_angle_cls=True, lang='fr', det_db_box_thresh=0.2, rec_algorithm="CRNN")

def extract_semantic_data(image_path: str, document_type: str):
    try:
        ocr_results = ocr.ocr(image_path, cls=True)
        raw_text = "\n".join([result[1][0] for result in ocr_results[0] if result[1][1] > 0.5])
        return generate_structured_data_with_llama2(raw_text, document_type)
    except Exception as e:
        return {"error": str(e)}
