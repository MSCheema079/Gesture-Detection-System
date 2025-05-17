from flask import Flask, request, jsonify
from Gesture import GestureRecognizer
import cv2
import numpy as np
import os

app = Flask(__name__)
recognizer = GestureRecognizer()

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Read the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Optionally get source_folder from request (e.g., form data)
    source_folder = request.form.get('source_folder', 'train01')  # Default to 'train01'
    
    # Detect gesture
    result = recognizer.detect_gesture(img, source_folder)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)