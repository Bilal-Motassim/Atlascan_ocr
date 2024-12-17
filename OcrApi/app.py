import os
import cv2
from PIL import Image
import pytesseract
from flask import Flask, request, jsonify
from flask_cors import CORS  # Importer Flask-CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/extract-text', methods=['POST'])
def extract_text_api():
    if 'image' not in request.files:
        return jsonify({"error": "Aucun fichier image fourni"}), 400

    # Sauvegarder l'image uploadée
    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    # Extraire le texte
    result = extract_text_to_json(image_path)
    os.remove(image_path)  # Nettoyage
    return jsonify(result)

def extract_text_to_json(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Image introuvable ou non valide"}

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer une binarisation
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Configurer pytesseract
    custom_config = r'--oem 3 --psm 6 -l ara+fra'
    extracted_text = pytesseract.image_to_string(Image.fromarray(binary_image), config=custom_config)

    # Séparer les informations en arabe et français
    arabic_text = []
    french_text = []
    for line in extracted_text.splitlines():
        if any('\u0600' <= char <= '\u06FF' for char in line):
            arabic_text.append(line.strip())
        else:
            french_text.append(line.strip())

    return {
        "arabic": arabic_text,
        "french": french_text
    }

if __name__ == '__main__':
    app.run(debug=True)
