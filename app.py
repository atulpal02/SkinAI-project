from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

# ---------------- CONFIG ---------------- #

# Static folder for uploaded images
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Model location on Hugging Face
MODEL_URL = "https://huggingface.co/atulpal02/skinai-ham10000-model/resolve/main/ham10000_model_final.h5"
MODEL_PATH = "model.h5"

# ✅ Automatically download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Model downloaded successfully.")

# ✅ Load the trained HAM10000 model
model = load_model(MODEL_PATH)

# Label mapping (same as your training order)
label_map = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

# Disease details dictionary
disease_info = {
    "akiec": {
        "name": "Actinic Keratoses",
        "symptoms": "Rough, scaly patches caused by chronic sun exposure.",
        "seriousness": "Moderate — can progress to skin cancer if untreated."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "symptoms": "Shiny bump or pink growth on sun-exposed skin.",
        "seriousness": "High — most common skin cancer; rarely spreads but requires treatment."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesions",
        "symptoms": "Warty, waxy lesions often confused with melanoma.",
        "seriousness": "Low — harmless but cosmetically concerning."
    },
    "df": {
        "name": "Dermatofibroma",
        "symptoms": "Firm red or brown bumps, usually on the legs.",
        "seriousness": "Low — benign and slow growing."
    },
    "mel": {
        "name": "Melanoma",
        "symptoms": "New or changing mole, irregular color/border.",
        "seriousness": "Severe — aggressive skin cancer; early detection essential."
    },
    "nv": {
        "name": "Melanocytic Nevi (Moles)",
        "symptoms": "Common brown/black moles on the skin.",
        "seriousness": "Low — monitor for shape or color changes."
    },
    "vasc": {
        "name": "Vascular Lesions",
        "symptoms": "Bright red or purple blood vessel marks.",
        "seriousness": "Low to Moderate — usually benign."
    }
}

# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template("base.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/consult')
def skin_health():
    return render_template("predict.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # For direct GET access
    if request.method == 'GET':
        return render_template("base.html")

    # Validate upload
    if 'image' not in request.files:
        return render_template("base.html", prediction_text="No image uploaded.")

    f = request.files['image']
    if f.filename == '':
        return render_template("base.html", prediction_text="Please select an image.")

    # Save uploaded image
    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)

    # Load & preprocess image
    img = load_img(filepath, target_size=(64, 64))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    label_idx = np.argmax(preds, axis=1)[0]
    predicted_class = label_map[label_idx]
    confidence = float(preds[0][label_idx]) * 100

    # Fetch disease info
    info = disease_info.get(predicted_class, None)

    # Render result
    return render_template("result.html", info=info, image_file=filename, confidence=confidence)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# import numpy as np
# import os
# from keras.models import load_model
# from keras.preprocessing import image

# app = Flask(__name__)

# # Load your trained model
# model = load_model("models/ham10000_model.h5")

# # Define the correct label order (same order used during training)
# # Make sure this matches your training data generator class order!
# label_map = {
#     0: "akiec",
#     1: "bcc",
#     2: "bkl",
#     3: "df",
#     4: "mel",
#     5: "nv",
#     6: "vasc"
# }

# # Disease information dictionary
# disease_info = {
#     "akiec": {
#         "name": "Actinic Keratoses",
#         "symptoms": "Rough, scaly patches on the skin caused by sun exposure.",
#         "seriousness": "Moderate — can develop into squamous cell carcinoma if untreated."
#     },
#     "bcc": {
#         "name": "Basal Cell Carcinoma",
#         "symptoms": "Small shiny bump or nodule, often on sun-exposed areas.",
#         "seriousness": "High — the most common form of skin cancer; rarely spreads but requires treatment."
#     },
#     "bkl": {
#         "name": "Benign Keratosis-like Lesions",
#         "symptoms": "Waxy, wart-like growths that are usually non-cancerous.",
#         "seriousness": "Low — harmless but may resemble melanoma."
#     },
#     "df": {
#         "name": "Dermatofibroma",
#         "symptoms": "Firm, raised bump on the skin, often on legs or arms.",
#         "seriousness": "Low — benign and generally not harmful."
#     },
#     "mel": {
#         "name": "Melanoma",
#         "symptoms": "New or changing mole with irregular borders or color.",
#         "seriousness": "Severe — aggressive skin cancer; early detection is critical."
#     },
#     "nv": {
#         "name": "Melanocytic Nevi (Moles)",
#         "symptoms": "Dark brown spots or moles, usually harmless.",
#         "seriousness": "Low — benign, but monitor for changes."
#     },
#     "vasc": {
#         "name": "Vascular Lesions",
#         "symptoms": "Red or purple spots caused by abnormal blood vessels.",
#         "seriousness": "Low to Moderate — typically benign but may require treatment."
#     }
# }

# @app.route('/')
# def index():
#     return render_template("base.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return render_template("base.html", prediction_text="No file uploaded.")

#     f = request.files['image']
#     if f.filename == '':
#         return render_template("base.html", prediction_text="Please select an image file.")

#     # Save uploaded image
#     basepath = os.path.dirname(__file__)
#     filepath = os.path.join(basepath, secure_filename(f.filename))
#     f.save(filepath)

#     # Preprocess the image
#     img = image.load_img(filepath, target_size=(64, 64))
#     x = image.img_to_array(img) / 255.0
#     x = np.expand_dims(x, axis=0)

#     # Make prediction
#     preds = model.predict(x)
#     label_idx = np.argmax(preds, axis=1)[0]
#     predicted_class = label_map[label_idx]

#     # Retrieve disease details
#     info = disease_info.get(predicted_class, None)

#     return render_template("result.html", info=info, image_file=f.filename)

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)

# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# import numpy as np
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Initialize Flask app
# app = Flask(__name__)

# # Load your trained CNN model
# model = load_model("ham10000_model_final.h5")

# # Define class labels (based on your label_map and CNN output)
# CLASS_LABELS = [
#     'Actinic keratoses',          # akiec
#     'Basal cell carcinoma',       # bcc
#     'Benign keratosis-like lesions',  # bkl
#     'Dermatofibroma',             # df
#     'Melanoma',                   # mel
#     'Melanocytic nevi',           # nv
#     'Vascular lesions'            # vasc
# ]

# @app.route('/')
# def index():
#     return render_template("base.html")  # Home page with upload form

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return render_template("base.html", prediction_text="No file uploaded.")
    
#     f = request.files['image']
#     if f.filename == '':
#         return render_template("base.html", prediction_text="Please select an image file.")

#     # Save uploaded file temporarily
#     basepath = os.path.dirname(__file__)
#     filepath = os.path.join(basepath, secure_filename(f.filename))
#     f.save(filepath)

#     # Preprocess image for model prediction
#     img = image.load_img(filepath, target_size=(64, 64))  # same as training
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = x / 255.0  # normalize just like training set

#     # Predict the class
#     preds = model.predict(x)
#     pred_class = np.argmax(preds, axis=1)[0]
#     result = f"The predicted skin lesion is: \"{CLASS_LABELS[pred_class]}\""

#     return render_template("base.html", prediction_text=result)

# if __name__ == "__main__":
#     app.run(debug=True,port = 5001)
