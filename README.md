# Implementasi Face Detection and Face Recognition - Workshop CV

## Deskripsi
Repository ini adalah project pengenalan wajah menggunakan metode **Eigenfaces** (PCA) yang dikombinasikan dengan **SVM classifier**.  
Dataset berisi gambar wajah dari 9 orang, masing-masing terdiri dari 20 gambar. Model yang sudah dilatih bisa digunakan untuk real-time face recognition menggunakan webcam.

---

## Struktur Folder
- `README.md` — Dokumentasi project.
- `train_model.py` — Code untuk melatih model Eigenfaces + SVM dari dataset gambar.
- `webcam_face_recognition.py` — Code untuk menjalankan face recognition secara real-time dengan webcam.
- `eigenface_pipeline.pkl` — File model hasil training.
- `images/` — Folder berisi dataset gambar wajah untuk training.
- `requirements.txt` — Daftar library Python yang diperlukan.
- `results/` — Folder berisi hasil testing, seperti screenshot atau video demo.
  
---

## Steps Menjalankan Code

1. Clone Repository Ini

```
git clone https://github.com/username/Face-Recognition-Project-Eigenfaces.git
cd Face-Recognition-Project-Eigenfaces
```
2. Install semua extension yang diperlukan untuk menjalankan code di VSCode
   
3. Setup Virtual Environment
```
python -m venv .venv
.venv/Scripts/activate
```
4. Install Requirements
```
pip install -r requirements.txt
```
5. Jalankan Script

**Penting:** Sebelum menjalankan `train_model.py`, ubah nilai variabel `dataset_dir` sesuai dengan lokasi folder images di komputer masing-masing.

Jika sudah melakukan langkah-langkah sebelumnya, kode dapat dijalankan. Jika ingin melatih ulang model:
```
python train_model.py

```

Jika ingin melakukan real-time face recognition:
```
python webcam_face_recognition.py

```
