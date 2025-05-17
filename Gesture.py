import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import load
import os

class GestureRecognizer:
    def __init__(self):
        self.model = load("models/rf_model.joblib")
        self.scaler = load("models/scaler.joblib")
        self.le = load("models/label_encoder.joblib")
    
    def preprocess_image(self, img):
        """Extract 120 features from image using a 6x10 grid"""
        img = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred, lower, upper)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = edges / 255.0
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        features = []
        rows, cols = 6, 10
        cell_height = 64 // rows
        cell_width = 64 // cols
        
        for i in range(rows):
            for j in range(cols):
                y1 = i * cell_height
                y2 = (i + 1) * cell_height
                x1 = j * cell_width
                x2 = (j + 1) * cell_width
                
                cell_edges = edges[y1:y2, x1:x2]
                cell_grad = grad_mag[y1:y2, x1:x2]
                mean_edge = np.mean(cell_edges) if cell_edges.size > 0 else 0
                var_edge = np.var(cell_edges) if mean_edge > 0 else 0
                grad_mean = np.mean(cell_grad) if cell_grad.size > 0 else 0
                features.extend([mean_edge, var_edge, grad_mean])
        
        features = features[:120]  # Ensure exactly 120 features
        print(f"Detection: Extracted {len(features)} features: {features}")
        return np.array(features)

    def detect_gesture(self, img, source_folder="train01"):
        """Detect gesture from image with optional source_folder"""
        features = self.preprocess_image(img)
        if len(features) != 120:
            raise ValueError(f"Expected 120 features, but got {len(features)}")
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        confidence = np.max(probabilities) * 100
        
        label = self.le.inverse_transform(prediction)[0]
        print(f"Debug: Predicted label: {label}, Confidence: {confidence:.2f}%")
        print(f"Debug: Feature values: {features}")
        print(f"Debug: Prediction probabilities: {probabilities}")
        
        return {
            "gesture": label,
            "confidence": f"{confidence:.2f}%",
            "features": features.tolist()
        }

# Example usage (for testing)
if __name__ == "__main__":
    recognizer = GestureRecognizer()
    img = cv2.imread("sample_image.jpg")
    result = recognizer.detect_gesture(img)
    print(result)