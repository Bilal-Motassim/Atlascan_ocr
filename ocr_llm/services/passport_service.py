from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from utils.ocr_utils import extract_semantic_data

router = APIRouter()

@router.post("/extract/")
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
