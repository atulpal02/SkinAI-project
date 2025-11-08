#  SkinAI — AI-Based Skin Disease Detection
<img width="844" height="545" alt="Screenshot 2025-11-08 at 9 54 51 PM" src="https://github.com/user-attachments/assets/ef7529ba-8673-4dd7-88f4-70d03da94f1e" />
<img width="844" height="549" alt="Screenshot 2025-11-08 at 9 55 21 PM" src="https://github.com/user-attachments/assets/7e381142-965d-42c7-afff-bcc9633302f9" />
<img width="844" height="549" alt="Screenshot 2025-11-08 at 9 55 47 PM" src="https://github.com/user-attachments/assets/7b082277-e365-491c-a58e-f8b4f5da1b16" />
<img width="844" height="549" alt="Screenshot 2025-11-08 at 9 56 02 PM" src="https://github.com/user-attachments/assets/03dbf613-12ff-468b-b173-feefbb34d501" />
<img width="844" height="549" alt="Screenshot 2025-11-08 at 9 56 18 PM" src="https://github.com/user-attachments/assets/2e570c7b-880c-4874-b61d-41fa0b74f717" />
<img width="844" height="549" alt="Screenshot 2025-11-08 at 9 56 29 PM" src="https://github.com/user-attachments/assets/f90fa70d-2940-4ff3-9901-6f69e1d013d6" />


This repository contains a deep learning project that uses a Convolutional Neural Network (CNN) to detect various skin diseases from images. The goal of this project is to assist healthcare professionals by providing a model that can classify skin diseases, enhancing accessibility to diagnostics through machine learning.

#  SkinAI — AI-Based Skin Disease Detection

This project uses a **Convolutional Neural Network (CNN)** trained on the **HAM10000 dermatology image dataset** to detect and classify common skin diseases.  
The goal is to support **early skin disease detection** and provide an AI-assisted diagnosis platform for dermatologists and patients.


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Access via Hugging Face](#model-access-via-hugging-face)
- [Limitations](#limitations)
- [License](#license)


## 🌍 Overview

SkinAI leverages a **deep learning model (CNN)** to classify skin lesions from images into multiple diagnostic categories such as melanoma, basal cell carcinoma, benign keratosis, and more.

Due to the sensitive and large-scale nature of dermatology image datasets, this project is designed to **assist clinicians** and **improve accessibility** to skin condition analysis, not to replace medical judgment.

The app is built using **Flask**, enabling users to upload skin images and receive **real-time AI predictions** with disease information and confidence scores.


## Dataset

**Dataset:** [HAM10000 ("Human Against Machine with 10000 training images")](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
**Size:** ~6 GB (10,000+ high-quality dermatoscopic images)  
**Classes:**  
1. Actinic Keratoses (akiec)  
2. Basal Cell Carcinoma (bcc)  
3. Benign Keratosis-like Lesions (bkl)  
4. Dermatofibroma (df)  
5. Melanoma (mel)  
6. Melanocytic Nevi (nv)  
7. Vascular Lesions (vasc)

**Preprocessing Steps:**
- Image resizing to (64 × 64)
- Normalization to 0–1 range  
- Data augmentation (rotation, zoom, flipping)
- Train-validation split (80:20)


## Model Architecture

A **custom CNN model** built with TensorFlow/Keras, optimized for balanced accuracy and efficiency.

### Key Layers:
- **Convolutional Layers:** Feature extraction  
- **Pooling Layers:** Down-sampling  
- **Dense Layers:** Classification decision  
- **Softmax Output Layer:** Probability distribution across 7 classes  

**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam  
**Final Accuracy:** ~85% on validation set  

##  Technologies Used

| Category | Tools/Frameworks |
|-----------|------------------|
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Web App | Flask |
| Model Hosting | Hugging Face Hub |
| Deployment | Render / Vercel |


##  Results

- **Accuracy:** ~85% on validation data  
- **Loss:** ~0.35  
- **Model File Size:** ~180 MB (`.h5`)  
- **Dataset:** HAM10000 (6 GB)

The CNN effectively identifies most skin lesion classes, achieving particularly high confidence for melanoma and benign nevi detection.


##  Installation

To run the project locally:

### 1️ Clone this repository

git clone https://github.com/atulpal02/SkinAI.git
cd SkinAI

python -m venv env
source env/bin/activate   # On macOS/Linux
env\Scripts\activate

pip install -r requirements.txt


##Usage

Option 1: Use Pretrained Model (via Hugging Face)

The trained CNN model (~180 MB) is too large for GitHub’s 100 MB limit.
So it’s hosted on Hugging Face and automatically downloaded at runtime.

When you run the Flask app, it will:

Download the .h5 model from Hugging Face (if not already cached)

Load it into memory only once (to save Render resources)

Run predictions on uploaded skin images

Start the app:

python app.py


Then open your browser at:

http://127.0.0.1:5000


Upload an image and get your prediction with confidence and disease info.


Option 2:  Train Your Own Model (Offline Mode)

If you don’t want to rely on Hugging Face (e.g., for offline/local use):

Download the HAM10000 dataset from Kaggle

Train the CNN using your own script (e.g., train.py)

Save the trained model locally:

model.save("model.h5")


Comment out the Hugging Face download block in app.py:

# MODEL_URL = "https://huggingface.co/atulpal02/skinai-ham10000-model/resolve/main/ham10000_model_final.h5"
# Model loading from URL


Run Flask with your local model:

python app.py

Model Access via Hugging Face

You can view or download the pretrained model here:

🔗 Hugging Face Model:
https://huggingface.co/atulpal02/skinai-ham10000-model

The Flask app automatically fetches this model at startup if not found locally.


## ⚠️ Limitations

Requires ~1 GB RAM during model loading (Render free tier may crash)

Internet access needed for Hugging Face model download

Intended for educational and research use, not clinical diagnosis

Accuracy depends on lighting, image clarity, and skin tone variations

📜 License

This project is open-sourced under the MIT License.
You’re free to use, modify, and distribute with attribution.

## Author

Atul Pal

GitHub: @atulpal02

Hugging Face: atulpal02

Project: SkinAI — AI-Based Skin Disease Detector







