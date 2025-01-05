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

# Initialize PaddleOCR with improved settings
ocr = PaddleOCR(use_angle_cls=True, lang='fr', det_db_box_thresh=0.2, rec_algorithm="CRNN")

# Setup logging for debugging and monitoring
logger.add("logs/ocr_service.log", rotation="10 MB", retention="7 days", level="DEBUG")

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

        # Perform OCR
        logger.info("Running PaddleOCR.")
        ocr_results = ocr.ocr(image, cls=True)

        # Extract raw text from OCR results with confidence score filtering
        raw_text = ""
        for result in ocr_results[0]:
            text, confidence = result[1]
            if confidence > 0.5:  # Confidence threshold to filter out low-quality detections
                raw_text += text + "\n"
        
        logger.info(f"OCR Extracted Text:\n{raw_text}")

        # Parse structured data
        data = parse_id_card(raw_text)
        logger.info(f"Parsed Structured Data: {data}")

        return {"structured_data": data}

    except Exception as e:
        logger.error(f"Error during data extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Parse the OCR text for structured data with improved regex
def parse_id_card(raw_text: str) -> Dict[str, str]:
    """
    Extract structured fields using regex patterns based on OCR output structure.
    """
    logger.info("Parsing OCR text for structured data.")
    
    # Split the text into lines for easier parsing
    lines = raw_text.splitlines()

    # Ignore the first few lines (non-relevant OCR output)
    lines = lines[4:]

    # Initialize extracted data
    extracted_data = {
        "Prénom": None,
        "Nom": None,
        "Date de naissance": None,
        "Lieu de naissance": None,
        "Num d'identité": None,
        "Valable jusqu'au": None,
    }

    # Improved parsing logic with adjusted regex matching
    for line in lines:
        line = line.strip()

        # Match for "Prénom" (HAMZA) - Usually a word that is a person's name
        if not extracted_data["Prénom"] and re.match(r"^[A-Za-zÀ-ÿ]+$", line):
            extracted_data["Prénom"] = line  # First match as Prénom
        
        # Match for "Nom" (ELBOUZIDI) - The second name-like entry
        elif not extracted_data["Nom"] and extracted_data["Prénom"] and re.match(r"^[A-Za-zÀ-ÿ]+$", line):
            extracted_data["Nom"] = line  # Second name match as Nom
        
        # Match for "Date de naissance" (DD.MM.YYYY format)
        elif not extracted_data["Date de naissance"] and re.match(r"\d{2}\.\d{2}\.\d{4}", line):
            extracted_data["Date de naissance"] = line
        
        # Match for "Lieu de naissance" (SAFI) - Location often after names
        elif not extracted_data["Lieu de naissance"] and re.match(r"^[A-Za-zÀ-ÿ]+$", line):
            extracted_data["Lieu de naissance"] = line
        
        # Match for "Num d'identité" (HH246780) - ID number pattern (alphanumeric)
        elif not extracted_data["Num d'identité"] and re.match(r"^[A-Za-z0-9]+$", line):
            # Ignore numbers starting with "CAN"
            if line.startswith("CAN"):
                continue
            extracted_data["Num d'identité"] = line
        
        # Match for "Valable jusqu'au" - date field, e.g., "Valable jusqu'au 26.05.2031"
        elif not extracted_data["Valable jusqu'au"] and re.search(r"Valablejusqu'au\s*(\d{2}\.\d{2}\.\d{4})", line):
            extracted_data["Valable jusqu'au"] = re.search(r"\d{2}\.\d{2}\.\d{4}", line).group()

    # Return structured data with corrected fields
    return extracted_data

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
