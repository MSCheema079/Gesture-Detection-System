import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import warnings
from sklearn.decomposition import PCA
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration Constants
CONFIG = {
    "dataset_path": r"C:\Users\Saad Cheema\Desktop\2023-CS-729 Muhammad Saad Cheema AI project\AI LAB\datase",
    "train_folders": ["train01", "train02"],
    "image_size": 64,
    "k_neighbors": 9,
    "model_paths": {
        'knn': "models/knn_model.joblib",
        'nb': "models/nb_model.joblib",
        'rf': "models/rf_model.joblib",
        'kmeans': "models/kmeans_model.joblib"
    },
    "label_encoder_path": "models/label_encoder.joblib",
    "scaler_path": "models/scaler.joblib",  # Added scaler path
    "debug_dir": "debug_preprocessed",
    "test_size": 0.2,
    "random_state": 42,
    "n_clusters": 4  # From your latest request
}

# Setup directories
os.makedirs(CONFIG['debug_dir'], exist_ok=True)
os.makedirs("models", exist_ok=True)

class GestureRecognizer:
    def __init__(self):
        self.X = None
        self.y = None
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {
            'knn': KNeighborsClassifier(n_neighbors=CONFIG['k_neighbors'], weights='distance'),
            'nb': GaussianNB(),
            'rf': RandomForestClassifier(
                n_estimators=100,
                random_state=CONFIG['random_state'],
                class_weight='balanced',
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'kmeans': KMeans(n_clusters=CONFIG['n_clusters'], init='k-means++', 
                           random_state=CONFIG['random_state'], n_init=50)
        }

    def preprocess_image(self, img, debug_name=None):
        """Extract 120 features from image using a 6x10 grid"""
        try:
            img = cv2.resize(img, (CONFIG['image_size'], CONFIG['image_size']))
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
            cell_height = CONFIG['image_size'] // rows
            cell_width = CONFIG['image_size'] // cols
            
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
            print(f"Training: Extracted {len(features)} features: {features}")
            
            if debug_name:
                cv2.imwrite(os.path.join(CONFIG['debug_dir'], debug_name), edges * 255)
            return np.array(features)
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None

    def load_dataset(self):
        """Load and augment dataset with flip augmentation, then balance classes"""
        X, y = [], []
        folder_labels = {folder: [] for folder in CONFIG['train_folders']}  # Track labels per folder
        
        # Load data
        for folder in CONFIG['train_folders']:
            folder_path = os.path.join(CONFIG['dataset_path'], folder)
            if not os.path.exists(folder_path):
                print(f"Folder {folder} not found at {folder_path}")
                continue
                
            for root, dirs, files in os.walk(folder_path):
                for dir_name in dirs:
                    class_path = os.path.join(root, dir_name)
                    for file in os.listdir(class_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(class_path, file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                features = self.preprocess_image(img, f"{folder}_{dir_name}_{file.split('.')[0]}.png")
                                if features is not None:
                                    X.append(features)
                                    y.append(f"{folder}_{dir_name}")
                                    folder_labels[folder].append(f"{folder}_{dir_name}")
                                flipped = cv2.flip(img, 1)
                                features_flipped = self.preprocess_image(flipped, f"{folder}_{dir_name}_flipped_{file.split('.')[0]}.png")
                                if features_flipped is not None:
                                    X.append(features_flipped)
                                    y.append(f"{folder}_{dir_name}")
                                    folder_labels[folder].append(f"{folder}_{dir_name}")
        
        # Log labels per folder
        for folder, labels in folder_labels.items():
            print(f"Labels in {folder}: {Counter(labels)}")
        
        # Check label consistency between folders
        train01_labels = set(folder_labels['train01'])
        train02_labels = set(folder_labels['train02'])
        print(f"Unique labels in train01: {train01_labels}")
        print(f"Unique labels in train02: {train02_labels}")
        
        # Encode labels
        y_encoded = self.le.fit_transform(y)
        print(f"Dataset loaded: {len(X)} samples, {len(np.unique(y_encoded))} classes")
        class_dist = Counter(self.le.inverse_transform(y_encoded))
        print(f"Class distribution before balancing: {class_dist}")
        print(f"Label encoder mapping: {dict(zip(self.le.classes_, range(len(self.le.classes_))))}")
        
        # Balance the dataset by subsampling the majority class
        X_balanced, y_balanced = [], []
        min_class_count = min(class_dist.values())
        class_indices = {label: [] for label in class_dist.keys()}
        
        for idx, label in enumerate(self.le.inverse_transform(y_encoded)):
            class_indices[label].append(idx)
        
        for label, indices in class_indices.items():
            # Subsample to match the minority class count
            sampled_indices = np.random.choice(indices, min_class_count, replace=False)
            for idx in sampled_indices:
                X_balanced.append(X[idx])
                y_balanced.append(y_encoded[idx])
        
        self.X = np.array(X_balanced)
        self.y = np.array(y_balanced)
        balanced_dist = Counter(self.le.inverse_transform(self.y))
        print(f"Class distribution after balancing: {balanced_dist}")
        
        dump(self.le, CONFIG['label_encoder_path'])
        print("Label encoder saved to:", CONFIG['label_encoder_path'])
        return self.X, self.y

    def train_models(self):
        """Train and evaluate all models"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state'],
            stratify=self.y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        dump(self.scaler, CONFIG['scaler_path'])
        print("Scaler saved to:", CONFIG['scaler_path'])
        
        results = {}
        for name in ['knn', 'nb', 'rf']:
            self.models[name].fit(X_train_scaled, y_train)
            y_pred = self.models[name].predict(X_test_scaled)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True),
                'matrix': confusion_matrix(y_test, y_pred)
            }
            self._save_model(name)
        
        self.models['kmeans'].fit(self.scaler.transform(self.X))
        self._save_model('kmeans')
        
        self._create_visualizations(results, X_test_scaled, y_test)
        return results

    def _save_model(self, model_name):
        dump(self.models[model_name], CONFIG['model_paths'][model_name])
        print(f"{model_name.upper()} model saved")

    def _create_visualizations(self, results, X_test, y_test):
        classes = self.le.classes_
        
        for name in ['knn', 'nb', 'rf']:
            self._plot_confusion_matrix(
                y_test, results[name]['matrix'], 
                classes, f"{name}_confusion.png"
            )
            self._plot_metrics(
                name, results[name]['report'], 
                results[name]['accuracy'], 
                f"{name}_metrics.png"
            )
        
        self._plot_kmeans_clusters(f"kmeans_clusters.png")
        self._plot_model_comparison(results, f"model_comparison.png")

    def _plot_confusion_matrix(self, y_true, cm, classes, filename):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(filename)
        plt.close()

    def _plot_metrics(self, model_name, report, accuracy, filename):
        metrics = ['precision', 'recall', 'f1-score']
        values = [report['weighted avg'][m] for m in metrics]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Accuracy'] + [m.capitalize() for m in metrics], 
                      [accuracy] + values,
                      color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
        plt.title(f"{model_name.upper()} Performance Metrics")
        plt.ylim(0, 1)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f"{height:.2f}", ha='center')
        plt.savefig(filename)
        plt.close()

    def _plot_kmeans_clusters(self, filename):
        X_scaled = self.scaler.transform(self.X)
        labels = self.models['kmeans'].labels_
        centroids = self.models['kmeans'].cluster_centers_
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        centroids_pca = pca.transform(centroids)
        
        silhouette_avg = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {silhouette_avg:.2f}")
        
        if silhouette_avg <= 0.3:
            print("Warning: Silhouette Score <= 0.3, clusters may not be well-separated.")
        
        plt.figure(figsize=(35, 30))  # Very large figure for maximum separation
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A6']  # 4 distinct colors
        markers = ['o', 's', '^', 'v']  # 4 distinct markers
        
        for i in range(CONFIG['n_clusters']):
            # Apply significant jitter for separation
            jitter_x = np.random.normal(0, 2.0, size=X_pca[labels == i, 0].shape)  # Increased to 2.0
            jitter_y = np.random.normal(0, 2.0, size=X_pca[labels == i, 1].shape)  # Increased to 2.0
            mask = labels == i
            if np.sum(mask) > 0:  # Ensure cluster has points
                plt.scatter(X_pca[mask, 0] + jitter_x, X_pca[mask, 1] + jitter_y,
                           c=colors[i], marker=markers[i], label=f'Cluster {i+1}',
                           alpha=0.3, s=250, edgecolors='black', linewidth=1)
        
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   marker='X', s=800, linewidths=3,
                   color='black', edgecolors='red',
                   label='Centroids', zorder=10)
        
        plt.title(f"{CONFIG['n_clusters']}-Cluster K-Means Visualization\n(Silhouette: {silhouette_avg:.2f})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=400, bbox_inches='tight')  # High resolution
        plt.close()

    def _plot_model_comparison(self, results, filename):
        metrics = ['precision', 'recall', 'f1-score']
        model_names = ['KNN', 'Naive Bayes', 'Random Forest']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics):
            values = [results[name]['report']['weighted avg'][metric] for name in ['knn', 'nb', 'rf']]
            plt.bar(np.arange(len(model_names)) + i*0.25, values,
                   width=0.25, color=colors[i], label=metric.capitalize())
        
        accuracies = [results[name]['accuracy'] for name in ['knn', 'nb', 'rf']]
        plt.plot(np.arange(len(model_names)) + 0.3, accuracies,
                color='#9b59b6', marker='o', linestyle='--',
                linewidth=2, markersize=8, label='Accuracy')
        
        plt.xticks(np.arange(len(model_names)) + 0.3, model_names)
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("Model Performance Comparison")
        plt.legend(loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    recognizer = GestureRecognizer()
    try:
        X, y = recognizer.load_dataset()
        if len(np.unique(y)) < 3:
            raise ValueError("Minimum 3 classes required for clustering")
        results = recognizer.train_models()
        print("\nTraining completed successfully!")
        print(f"Models saved to: {CONFIG['model_paths']}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Troubleshooting:")
        print("1. Verify dataset path exists")
        print("2. Check image file formats (.jpg, .jpeg, .png)")
        print("3. Ensure minimum 3 gesture classes")
        print("4. Check class distribution above")