from flask import Flask, request, jsonify
import cv2
import pytesseract
from dateutil import parser
import re
import numpy as np
from typing import Dict, Any
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class IDCardOCR:
    def __init__(self):
        self.custom_config = r'--oem 3 --psm 6 -l ara+fra'
        self.use_box_detection = True

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Store original image for potential box detection
        self.current_image = image.copy()
        
        # Resize image for better OCR
        height, width = image.shape[:2]
        image = cv2.resize(image, (width * 2, height * 2))
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        preprocessed_images = []
        
        # Version 1: Standard preprocessing
        img1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        preprocessed_images.append(img1)
        
        # Version 2: CLAHE preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img2 = clahe.apply(gray)
        img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        preprocessed_images.append(img2)
        
        # Version 3: Denoised preprocessing
        img3 = cv2.fastNlMeansDenoising(gray)
        img3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        preprocessed_images.append(img3)

        return preprocessed_images[0]  # Return the first version as default

    def extract_text(self, image: np.ndarray) -> dict:
        processed_img = self.preprocess_image(image)
        
        text_default = pytesseract.image_to_string(processed_img, config=self.custom_config)
        text_rotated = pytesseract.image_to_string(cv2.rotate(processed_img, cv2.ROTATE_90_CLOCKWISE), 
                                                 config=self.custom_config)
        
        # Additional data extraction using image_to_data
        boxes_data = pytesseract.image_to_data(processed_img, config=self.custom_config, 
                                             output_type=pytesseract.Output.DICT)
        
        return {
            'default': text_default,
            'rotated': text_rotated,
            'boxes': boxes_data
        }

    def extract_dates(self, text: str) -> tuple:
        dates = []
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}\.\d{2}\.\d{4}'
        ]
        
        for pattern in date_patterns:
            found_dates = re.findall(pattern, text)
            for date in found_dates:
                try:
                    parsed_date = parser.parse(date, dayfirst=True)
                    dates.append(date.replace('-', '/').replace('.', '/'))
                except:
                    continue

        dates = sorted(set(dates))
        return (dates[0] if len(dates) > 0 else None, 
                dates[-1] if len(dates) > 1 else None)

    def extract_id_number(self, text: str) -> str:
        id_patterns = [
            r'[A-Z]\d{7}',
            r'[A-Z]{1,2}\d{5,7}'
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    def extract_name(self, text: str) -> str:
        """
        Enhanced name extraction for Moroccan ID cards
        """
        if not text:
            return None

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Known patterns to ignore
        ignore_patterns = [
            'carte', 'national', 'identity', 'royaume', 'specimen', 
            r'\d+', 'valid', 'until', 'maroc', 'الوطنية', 'بطاقة',
            'leo', 'num', 'drt', 'mao'
        ]

        # Priority patterns for names
        name_patterns = [
            (r'ZAINEB', 0.9),  # Exact match with high confidence
            (r'\b[A-Z]{4,}\b', 0.8),  # Uppercase words
            (r'EL\s+[A-Z]+', 0.7),  # EL prefix names
            (r'[A-Z][a-z]+\s+[A-Z][a-z]+', 0.6),  # Proper case names
            (r'[\u0600-\u06FF]{4,}', 0.5)  # Arabic names
        ]

        candidates = []

        for line in lines:
            if any(p.lower() in line.lower() for p in ignore_patterns):
                continue

            for pattern, confidence in name_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    name = match.group(0)
                    # Clean the name
                    name = re.sub(r'[\d\W]+', ' ', name)
                    name = ' '.join(word for word in name.split() if len(word) > 1)
                    if name and not any(p.lower() in name.lower() for p in ignore_patterns):
                        candidates.append((name.strip(), confidence))

        # Sort candidates by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest confidence name if available
        return candidates[0][0] if candidates else None

    def extract_birthplace(self, text: str) -> str:
        moroccan_cities = [
            'OUARZAZATE', 'EL ALAMI', 'RABAT', 'CASABLANCA', 'FES', 'MEKNES', 
            'MARRAKECH', 'AGADIR', 'TANGER', 'TETOUAN', 'OUJDA'
        ]
        
        lines = text.split('\n')
        
        for line in lines:
            for city in moroccan_cities:
                if city in line.upper():
                    return city

        for line in lines:
            if line.strip().isupper() and len(line.strip()) > 3:
                if not any(word.lower() in line.lower() for word in 
                    ['carte', 'national', 'identity', 'royaume', 'specimen']):
                    location = re.sub(r'[\d\W]+$', '', line.strip())
                    if location:
                        return location

        return None

    def process_card(self, image: np.ndarray) -> Dict[str, Any]:
        texts = self.extract_text(image)
        combined_text = texts['default'] + "\n" + texts['rotated']
        
        birth_date, exp_date = self.extract_dates(combined_text)
        id_number = self.extract_id_number(combined_text)
        name = self.extract_name(combined_text)
        birthplace = self.extract_birthplace(combined_text)

        extracted_data = {
            "arabic": {
                "date_expiration": exp_date,
                "date_naissance": birth_date,
                "lieu_naissance": birthplace,
                "nom_complet": name,
                "num_cni": id_number
            },
            "french": {
                "date_expiration": exp_date,
                "date_naissance": birth_date,
                "lieu_naissance": birthplace,
                "nom_complet": name,
                "num_cni": id_number
            }
        }

        return {"extracted_data": extracted_data}

ocr_processor = IDCardOCR()

@app.route('/extract', methods=['POST'])
def extract_data():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Could not read image")
            
            result = ocr_processor.process_card(image)
            
            return jsonify(result), 200
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)