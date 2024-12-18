import os
import re
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Retrieve the API key from the environment
API_KEY = os.getenv('OCR_API_KEY')

@app.route('/extract', methods=['POST'])
def extract_info():
    """
    Endpoint to handle the image upload and extract information.
    """
    # Validate API Key
    client_api_key = request.headers.get('x-api-key')
    if not client_api_key or client_api_key != API_KEY:
        return jsonify({'error': 'Unauthorized - Invalid API Key'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Preprocess image
        preprocessed_img = preprocess_image(filepath)

        # Extract text
        extracted_text = pytesseract.image_to_string(preprocessed_img, lang='ara+fra', config='--psm 6')

        # Process extracted text
        structured_data = process_extracted_text(extracted_text)

        return jsonify({"extracted_data": structured_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        os.remove(filepath)

def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy.
    - Convert to grayscale
    - Apply adaptive thresholding
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(binary)

def process_extracted_text(text):
    """
    Dynamically process text to extract structured fields in Arabic and French.
    """
    data = {
        "arabic": {"nom_complet": None, "date_naissance": None, "lieu_naissance": None, "num_cni": None, "date_expiration": None},
        "french": {"nom_complet": None, "date_naissance": None, "lieu_naissance": None, "num_cni": None, "date_expiration": None}
    }

    lines = [clean_text(line) for line in text.split("\n") if clean_text(line)]

    # Variables to accumulate names and handle other fields
    french_name = []
    arabic_name = []
    arabic_lieu_found = False  # Track Arabic place of birth

    for i, line in enumerate(lines):
        # Extract dates
        date_match = re.search(r"\d{2}\.\d{2}\.\d{4}", line)
        if date_match:
            if "Valable" in line or "صالحة" in line:
                data["french"]["date_expiration"] = date_match.group()
                data["arabic"]["date_expiration"] = date_match.group()
            else:
                data["french"]["date_naissance"] = date_match.group()
                data["arabic"]["date_naissance"] = date_match.group()

        # Extract ID number
        id_match = re.search(r"\b[A-Z]\d{6,}\b", line)
        if id_match:
            card_number = id_match.group()
            data["french"]["num_cni"] = card_number
            data["arabic"]["num_cni"] = card_number

        # Extract names
        if re.search(r"^[A-Z\s]+$", line):  # French names
            french_name.append(line.strip())
        elif re.search(r"^[\u0600-\u06FF\s]+$", line):  # Arabic names
            arabic_name.append(line.strip())

        # Extract Arabic "lieu_naissance"
        if not arabic_lieu_found and "ب" in line and not re.search(r"تاريخ", line):
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if re.match(r"^[\u0600-\u06FF\s]+$", next_line):
                data["arabic"]["lieu_naissance"] = next_line.strip()
                arabic_lieu_found = True

        # Extract French "lieu_naissance"
        if "à" in line:
            data["french"]["lieu_naissance"] = line.split("à")[-1].strip()

    # Combine names
    data["french"]["nom_complet"] = " ".join(french_name) if french_name else None
    data["arabic"]["nom_complet"] = " ".join(arabic_name) if arabic_name else None

    return data

def clean_text(text):
    """
    Clean text by removing diacritics, control characters, and extra spaces.
    """
    text = text.replace("\u200f", "").replace("\u200e", "")  # Remove control characters
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", text)  # Remove Arabic diacritics
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s\.\-]", "", text)  # Allow Arabic, French, and numbers
    return re.sub(r"\s+", " ", text).strip()  # Normalize spaces

if __name__ == '__main__':
    app.run(debug=True)
