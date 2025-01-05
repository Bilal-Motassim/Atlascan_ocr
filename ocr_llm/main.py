from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
import cv2
import numpy as np
import json
from loguru import logger
import ollama

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang='fr', det_db_box_thresh=0.2, rec_algorithm="CRNN")

logger.add("logs/ocr_service.log", rotation="10 MB", retention="7 days", level="DEBUG")


def clean_ocr_text(raw_text: str) -> str:
    """
    Clean OCR text to remove unwanted patterns.
    """
    lines = raw_text.splitlines()
    cleaned_lines = [line for line in lines if not line.startswith("CAN")]
    return "\n".join(cleaned_lines)


def generate_structured_data_with_llama2(raw_text: str, document_type: str) -> Dict:
    """
    Process raw text using Llama 2 for better structured data extraction.
    """
    try:
        logger.info(f"Generating structured data using Llama 2 for {document_type}...")

        raw_text = clean_ocr_text(raw_text)

        if document_type == "id_card":
            prompt = f"""
            Extract structured data from the following ID card text. 
            Always return only a valid JSON and no explanations.
            
            {raw_text}
            
            Ensure the JSON structure exactly matches:
            {{
                "Prénom": "string",
                "Nom de famille": "string",
                "Date de naissance": "date",
                "Lieu de naissance": "string",
                "Num d'identité": "string",
                "Valable jusqu'au": "date"
            }}
            """
        elif document_type == "passport":
            prompt = f"""
            Extract structured data from the following passport text. 
            Always return only a valid JSON and no explanations.
            
            {raw_text}
            
            Ensure the JSON structure exactly matches:
            {{
                "Nom": "string",
                "Prénom": "string",
                "Date de naissance": "date",
                "Nationalité": "string",
                "Numéro de passeport": "string",
                "Date d'expiration": "date"
            }}
            """
        elif document_type == "drivers_license":
            prompt = f"""
            Extract structured data from the following driver's license text. 
            Always return only a valid JSON and no explanations.
            
            {raw_text}
            
            Ensure the JSON structure exactly matches:
            {{
                "Nom": "string",
                "Prénom": "string",
                "Date de naissance": "date",
                "Numéro de permis": "string",
                "Date d'expiration": "date",
                "Adresse": "string"
            }}
            """
        else:
            raise ValueError("Unknown document type")

        response = ollama.chat(model="llama2:latest", messages=[{"role": "user", "content": prompt}])
        structured_data_content = response.get('message', {}).get('content', '').strip()

        if not structured_data_content:
            logger.error(f"Empty response for {document_type} from Llama 2.")
            return {"error": f"Empty response from Llama 2 for {document_type}"}

        try:
            structured_data = json.loads(structured_data_content)
            logger.info(f"Llama 2 generated data for {document_type}: {structured_data}")
            return structured_data
        except json.JSONDecodeError:
            logger.error(f"JSON decoding failed for {document_type}. Reattempting with fallback.")
            return {"error": "Llama 2 did not return valid JSON"}

    except Exception as e:
        logger.error(f"Error using Llama 2 for {document_type}: {str(e)}")
        return {"error": f"Failed to process text with Llama 2 for {document_type}"}


def extract_semantic_data(image_path: str, document_type: str) -> Dict:
    """
    Perform OCR on the input image and extract structured data using PaddleOCR and Llama 2.
    """
    try:
        logger.info(f"Running OCR on the input image for {document_type}...")

        ocr_results = ocr.ocr(image_path, cls=True)

        raw_text = ""
        for result in ocr_results[0]:
            text, confidence = result[1]
            if confidence > 0.5: 
                raw_text += text + "\n"

        logger.info(f"Raw OCR Text for {document_type}:\n{raw_text}")

        structured_data = generate_structured_data_with_llama2(raw_text, document_type)
        return structured_data

    except Exception as e:
        logger.error(f"Error during semantic data extraction for {document_type}: {str(e)}")
        return {"error": f"Failed to extract semantic data from the image for {document_type}"}


def extract_semantic_data(image_path: str, document_type: str) -> Dict:
    """
    Perform OCR on the input image and extract structured data using PaddleOCR and Llama 2.
    """
    try:
        logger.info(f"Running OCR on the input image for {document_type}...")
        
        ocr_results = ocr.ocr(image_path, cls=True)

        raw_text = ""
        for result in ocr_results[0]:
            text, confidence = result[1]
            if confidence > 0.5:  
                raw_text += text + "\n"

        logger.info(f"Raw OCR Text for {document_type}:\n{raw_text}")

        structured_data = generate_structured_data_with_llama2(raw_text, document_type)
        return structured_data

    except Exception as e:
        logger.error(f"Error during semantic data extraction for {document_type}: {str(e)}")
        return {"error": f"Failed to extract semantic data from the image for {document_type}"}


@app.post("/extract-id-card-data/")
async def extract_id_card_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")
        
        temp_image_path = "temp_id_card_image.jpg"
        cv2.imwrite(temp_image_path, image)

        structured_data = extract_semantic_data(temp_image_path, "id_card")
        return {"structured_data": structured_data}

    except Exception as e:
        logger.error(f"Error during API processing for ID card: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/extract-passport-data/")
async def extract_passport_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")
        
        temp_image_path = "temp_passport_image.jpg"
        cv2.imwrite(temp_image_path, image)

        structured_data = extract_semantic_data(temp_image_path, "passport")
        return {"structured_data": structured_data}

    except Exception as e:
        logger.error(f"Error during API processing for Passport: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/extract-drivers-license-data/")
async def extract_drivers_license_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")
        
        temp_image_path = "temp_drivers_license_image.jpg"
        cv2.imwrite(temp_image_path, image)

        structured_data = extract_semantic_data(temp_image_path, "drivers_license")
        return {"structured_data": structured_data}

    except Exception as e:
        logger.error(f"Error during API processing for Driver's License: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
