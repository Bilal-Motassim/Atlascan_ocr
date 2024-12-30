from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import logging
from loguru import logger
from typing import Dict

# Initialize FastAPI app
app = FastAPI()

# Initialize PaddleOCR with advanced settings
ocr = PaddleOCR(use_angle_cls=True, lang='fr', det_db_box_thresh=0.2, rec_algorithm="CRNN")

# Setup logging for debugging and monitoring
logger.add("logs/ocr_service.log", rotation="10 MB", retention="7 days", level="DEBUG")

# Preprocess the uploaded image
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for OCR:
    - Convert to grayscale.
    - Apply histogram equalization for better text visibility.
    """
    logger.info("Preprocessing image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

# Perform OCR and extract structured data
@app.post("/extract-data/")
async def extract_data(file: UploadFile = File(...)):
    try:
        # Read and load the image
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        logger.info("Image successfully loaded.")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Perform OCR
        logger.info("Running PaddleOCR.")
        ocr_results = ocr.ocr(processed_image, cls=True)

        # Extract raw text from OCR results
        raw_text = "\n".join([line[1][0] for line in ocr_results[0]])
        logger.info(f"OCR Extracted Text:\n{raw_text}")

        # Parse structured data
        data = parse_id_card(raw_text)
        logger.info(f"Parsed Structured Data: {data}")

        return {"structured_data": data}

    except Exception as e:
        logger.error(f"Error during data extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Parse the OCR text for structured data
def parse_id_card(raw_text: str) -> Dict[str, str]:
    """
    Extract structured fields using regex patterns based on OCR output structure.
    """
    logger.info("Parsing OCR text for structured data.")
    
    # Split the text into lines for easier parsing
    lines = raw_text.splitlines()
    
    # Initialize extracted data
    extracted_data = {
        "Prénom": None,
        "Nom": None,
        "Date de naissance": None,
        "Lieu de naissance": None,
        "Num d'identité": None,
        "Valable jusqu'au": None,
    }
    
    # Custom logic to match specific lines
    for line in lines:
        line = line.strip()
        if not extracted_data["Prénom"] and re.match(r"^[A-Z]+$", line):
            extracted_data["Prénom"] = line  # First uppercase name assumed as Prénom
        elif not extracted_data["Nom"] and re.match(r"^[A-Z]+$", line):
            extracted_data["Nom"] = line  # Next uppercase name assumed as Nom
        elif not extracted_data["Date de naissance"] and re.match(r"\d{2}\.\d{2}\.\d{4}", line):
            extracted_data["Date de naissance"] = line
        elif not extracted_data["Lieu de naissance"] and re.match(r"^[A-Z]+$", line):
            extracted_data["Lieu de naissance"] = line
        elif not extracted_data["Num d'identité"] and re.match(r"^[A-Z0-9]+$", line):
            extracted_data["Num d'identité"] = line
        elif not extracted_data["Valable jusqu'au"] and re.match(r"Valablejusqu'au\d{2}\.\d{2}\.\d{4}", line):
            extracted_data["Valable jusqu'au"] = re.search(r"\d{2}\.\d{2}\.\d{4}", line).group()

    # Validate extracted data
    validate_structured_data(extracted_data)

    return extracted_data

# Validate the extracted data
def validate_structured_data(data: Dict[str, str]) -> None:
    """
    Ensure critical fields are not missing; raise warnings for missing fields.
    """
    logger.info("Validating extracted structured data.")
    critical_fields = ["Prénom", "Nom", "Num d'identité"]
    for field in critical_fields:
        if data[field] == "Not Detected":
            logger.warning(f"Critical field '{field}' is missing.")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
