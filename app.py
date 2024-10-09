from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from PIL import Image
import io
import base64
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import easyocr
from typing import Dict, Any
import numpy as np
import time
from pyngrok import ngrok
import os

app = Flask(__name__, static_url_path='/assets', static_folder='assets')
CORS(app, resources={r"/*": {"origins": "*"}})

# Load M2M100 model and tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Add more languages as needed

# Start ngrok
ngrok.set_auth_token("2mvdbKaN0WGhsWJLNR6dja75qDb_4C2an6GPPfpZEmZHQ91sW")  # Optional: Set your ngrok auth token if you have one
public_url = ngrok.connect("5000")  # Expose your Flask app on port 5000 as a string
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000\"".format(public_url))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    data: Dict[str, Any] = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Perform OCR with EasyOCR
    result = reader.readtext(np.array(image))
    text = ' '.join([item[1] for item in result])
    
    return jsonify({'text': text})

@app.route('/translate', methods=['POST'])
def translate_text():
    data: Dict[str, Any] = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    text = data.get('text')
    source_lang = data.get('source_lang')
    target_lang = data.get('target_lang')

    if not all([text, source_lang, target_lang]):
        return jsonify({"error": "Missing required fields"}), 400

    # Simulate long-running translation process
    total_steps = 10  # Example total steps for progress
    for step in range(total_steps):
        time.sleep(0.5)  # Simulate work being done
        # Here you would update the progress in a real application

    translated_text = translate(text, source_lang, target_lang)
    
    return jsonify({'translated': translated_text})

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

if __name__ == '__main__':
    app.run(debug=True)