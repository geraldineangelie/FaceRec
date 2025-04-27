# Face-Recognition-Project-Eigenfaces

## Description
This project is an assignment for the Computer Vision course. It involves a dataset containing face images of 3 individuals, with at least 20 images per person. The dataset is used to train a face recognition model using the Eigenface method combined with a Support Vector Machine (SVM) classifier.

## Project Files
- `README.md` — Program guide.
- `train_model.py` — Python script to train a face recognition model.
- `webcam_face_recognition.py` — Python script for real-time face recognition using webcam.
- `eigenface_pipeline.pkl` — Trained model file using PCA (Principal Component Analysis) and SVM.
- `images/` — Folder containing the face image dataset used for training.
- `requirements.txt` — List of Python libraries required to run the project.
- `results/` — Folder containing screenshots and demo video of real-time recognition.

## Step-by-Step Instructions
Here are the steps to run the code using the existing trained model:

1. **Clone this repository**

```bash
git clone https://github.com/your-username/Face-Recognition-Project-Eigenfaces.git
cd Face-Recognition-Project-Eigenfaces

2. **Set-Up Virtual Environment**
```bash
python -m venv .venv
# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On MacOS/Linux:
source .venv/bin/activate

3. **Install all required libraries**
```bash
pip install -r requirements.txt

4. **Run real-time face recognition using webcam**
```bash
python webcam_face_recognition.py
