import cv2
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# ==== CONFIGURATION ====
face_size = (128, 128)
probability_threshold = 0.6  # Threshold minimal untuk menampilkan prediksi

# ==== CUSTOM CLASS ====
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_face

# ==== LOAD MODEL ====
with open('eigenface_pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# ==== HELPER FUNCTIONS ====

def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    return face_cascade.detectMultiScale(image_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

def crop_faces(image_gray, faces, return_all=True):
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            h_margin, w_margin = int(0.1 * h), int(0.1 * w)
            y1 = max(0, y - h_margin)
            y2 = min(image_gray.shape[0], y + h + h_margin)
            x1 = max(0, x - w_margin)
            x2 = min(image_gray.shape[1], x + w + w_margin)
            cropped_faces.append(image_gray[y1:y2, x1:x2])
            selected_faces.append((x, y, w, h))
    return cropped_faces, selected_faces

def resize_and_flatten(face):
    face_resized = cv2.resize(face, face_size)
    face_normalized = face_resized.astype(np.float32) / 255.0
    return face_normalized.flatten()

def eigenface_prediction(image_gray):
    faces = detect_faces(image_gray)
    cropped_faces, selected_faces = crop_faces(image_gray, faces, return_all=True)

    if len(cropped_faces) == 0:
        return [], [], []

    X_faces = np.array([resize_and_flatten(face) for face in cropped_faces])
    probs = np.max(pipe.predict_proba(X_faces), axis=1)
    predictions = pipe.predict(X_faces)

    return probs, predictions, selected_faces

def draw_text(image, label, prob, pos=(0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (0, 0, 0)
    text_color_bg = (0, 255, 0)

    x, y = pos
    prob_text = f'Prob: {prob:.2f}'
    (w1, h1), _ = cv2.getTextSize(prob_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y - h1 - h2 - 25), (x + max(w1, w2) + 20, y), text_color_bg, -1)
    cv2.putText(image, label, (x + 10, y - 10), font, font_scale, text_color, font_thickness)
    cv2.putText(image, prob_text, (x + 10, y - h2 - 15), font, font_scale, font_thickness)

def draw_result(image, probs, labels, coords):
    result_image = image.copy()
    for (x, y, w, h), label, prob in zip(coords, labels, probs):
        if prob >= probability_threshold:  # Hanya tampilkan kalau confidence cukup
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_text(result_image, label, prob, pos=(x, y))
    return result_image

# ==== MAIN LOOP ====

def webcam_recognition():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam tidak terbuka.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        probs, labels, coords = eigenface_prediction(gray)
        frame_with_result = draw_result(frame, probs, labels, coords)

        cv2.imshow("Webcam Face Recognition", frame_with_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_recognition()
