import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# === ARGPARSE, agar kode mudah dijalankan di local lain ===
parser = argparse.ArgumentParser(description='Train face recognition model.')
parser.add_argument('--dataset_dir', type=str, default='C:/FaceRecognition/images',
                    help='Path ke folder dataset images')
args = parser.parse_args()

dataset_dir = args.dataset_dir


# ==== CONFIGURATION ====
face_size = (128, 128)
random_state = 177

# ==== FUNCTIONS ====
# Load image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Failed to load {image_path}")
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return image, gray

# Detect face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
def detect_faces(image_gray, scale_factor=1.08, min_neighbors=5, min_size=(30, 30)):
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces

# Crop faces
def crop_faces(image_gray, faces, return_all=False):
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        if return_all:
            for (x, y, w, h) in faces:
                x_margin = int(w * 0.1)
                y_margin = int(h * 0.1)
                x_start = max(0, x - x_margin)
                y_start = max(0, y - y_margin)
                x_end = min(image_gray.shape[1], x + w + x_margin)
                y_end = min(image_gray.shape[0], y + h + y_margin)
                cropped_faces.append(image_gray[y_start:y_end, x_start:x_end])
                selected_faces.append((x, y, w, h))
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
            x_margin = int(w * 0.1)
            y_margin = int(h * 0.1)
            x_start = max(0, x - x_margin)
            y_start = max(0, y - y_margin)
            x_end = min(image_gray.shape[1], x + w + x_margin)
            y_end = min(image_gray.shape[0], y + h + y_margin)
            cropped_faces.append(image_gray[y:y+h, x:x+w])
            selected_faces.append((x, y, w, h))
    return cropped_faces, selected_faces

# Resize and flatten
def resize_and_flatten(face):
    face_resized = cv2.resize(face, face_size)
    face_normalized = cv2.normalize(face_resized, None, 0, 255, cv2.NORM_MINMAX)
    face_normalized = face_normalized.astype(np.float32) / 255.0
    return face_normalized.flatten()

# Data augmentation
def augment_face(face):
    augmented_faces = [face]

    # Flip
    augmented_faces.append(cv2.flip(face, 1))

    # Rotate
    for angle in [-10, -5, 5, 5, 10]:
        h, w = face.shape
        matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(face, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented_faces.append(rotated)

    # Brightness
    for beta in [-30, -15, 15, 30]:
        bright = np.clip(face.astype(np.int16) + beta, 0, 255).astype(np.uint8)
        augmented_faces.append(bright)

    # Contrast Variation
    for alpha in [0.8, 1.2]:
        contrast = np.clip(face.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        augmented_faces.append(contrast)

    # Translation
    for shift_x, shift_y in [(10,0), (-10,0), (0,10), (0,-10)]:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(face, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented_faces.append(shifted)

    # Gaussian Blur
    for ksize in [(3,3), (5,5)]:
        blurred = cv2.GaussianBlur(face, ksize, 0)
        augmented_faces.append(blurred)

    # Add Noise
    for var in [5, 10]:
        noise = np.copy(face).astype(np.float32)
        noise += np.random.normal(0, var, face.shape)
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        augmented_faces.append(noise)

    return augmented_faces

# Custom transformer
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_face

# ==== LOAD DATA ====

images = []
labels = []
label_counts = {}

for root, dirs, files in os.walk(dataset_dir):
    if root == dataset_dir:
        continue
    label = os.path.basename(root)
    label_counts[label] = 0
    for file in files:
        img_path = os.path.join(root, file)
        image, gray = load_image(img_path)
        if gray is None:
            continue
        faces = detect_faces(gray)
        if len(faces) == 0:
            continue
        cropped_faces, _ = crop_faces(gray, faces)
        for augmented_face in augment_face(cropped_faces[0]):
            flattened = resize_and_flatten(augmented_face)
            images.append(flattened)
            labels.append(label)
            label_counts[label] += 1

X = np.array(images)
y = np.array(labels)

print(f"[Total samples after augmentation: {len(X)}]")

# ==== SPLIT DATA ====

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state, stratify=y)

# ==== MODEL PIPELINE ====

pipe = Pipeline([
    ('centering', MeanCentering()),
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=random_state)),
    ('svc', SVC(probability=True, random_state=random_state))
])

param_grid = {
    'pca__n_components': [25, 40, 50, 75],
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 0.01, 0.001],
    'svc__kernel': ['rbf']
}

print("[Melakukan Grid Search]")
grid = GridSearchCV(pipe, param_grid, cv=3, verbose=1, n_jobs=2)
grid.fit(X_train, y_train)

print(f"[Best params: {grid.best_params_}]")

# ==== EVALUATE ====

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("[Classification Report:]")
print(classification_report(y_test, y_pred))

# ==== SAVE MODEL ====

with open('eigenface_pipeline.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("[Model saved as eigenface_pipeline.pkl]")

# ==== HELPER FUNCTIONS FOR INFERENCE ====

def eigenface_prediction(model, image_gray):
    faces = detect_faces(image_gray)
    cropped_faces, selected_faces = crop_faces(image_gray, faces, return_all=True)
    if len(cropped_faces) == 0:
        return None, None, None

    X_faces = []
    for face in cropped_faces:
        flattened = resize_and_flatten(face)
        X_faces.append(flattened)

    X_faces = np.array(X_faces)
    predictions = model.predict(X_faces)
    probs = np.max(model.predict_proba(X_faces), axis=1)
    return probs, predictions, selected_faces

def draw_text(image, label, score, pos=(0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (0, 0, 0)
    text_color_bg = (0, 255, 0)

    x, y = pos
    score_text = f'Score: {score:.2f}'
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y-h1-h2-25), (x+max(w1, w2)+20, y), text_color_bg, -1)
    cv2.putText(image, label, (x+10, y-10), font, font_scale, text_color, font_thickness)
    cv2.putText(image, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness)

def draw_result(image, scores, labels, coords):
    result_image = image.copy()
    for (x, y, w, h), label, score in zip(coords, labels, scores):
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        draw_text(result_image, label, score, pos=(x, y))
    return result_image

# ==== VISUALISASI EIGENFACES ====

print("[Menampilkan eigenfaces]")

pca = best_model.named_steps['pca']
n_components = pca.n_components_
eigenfaces = pca.components_.reshape((n_components, face_size[0], face_size[1]))

ncol = 4
nrow = (n_components + ncol - 1) // ncol

fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.5*nrow), subplot_kw={'xticks':[], 'yticks':[]})

for i, ax in enumerate(axes.flat):
    if i < n_components:
        ax.imshow(eigenfaces[i], cmap='gray')
        ax.set_title(f'Eigenface {i+1}')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

