import os
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from utils.preprocess import preprocess_image  # your preprocessing util

# Initialize Flask app
app = Flask(__name__)

# Upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load new trained model
MODEL_PATH = "model/final_vehicle_damage_model.keras"

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define class names (must match your training dataset folder structure)
CLASS_NAMES = ['Minor Damage', 'Moderate Damage', 'Severe Damage']

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home route
@app.route('/')
def index():
    return render_template('upload.html')


# Upload + Predict route
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('upload.html', error="Image is not provided")

    file = request.files['image']
    if file.filename == '':
        return render_template('upload.html', error="No file selected")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if model is None:
            return render_template('upload.html', error="Model not available. Please check server logs.")

        try:
            # Preprocess and predict
            img = preprocess_image(filepath)  # shape -> (1, 224, 224, 3)
            preds = model.predict(img)
            predicted_index = np.argmax(preds)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(np.max(preds))

            print(f"🧠 Prediction: {predicted_class} ({confidence*100:.2f}%)")

            return render_template(
                'result.html',
                damage=predicted_class,
                confidence=confidence,
                filename=filename
            )

        except Exception as e:
            return render_template('upload.html', error=f"Prediction error: {str(e)}")

    else:
        return render_template('upload.html', error="Invalid file type. Allowed: png, jpg, jpeg, gif")


if __name__ == '__main__':
    app.run(debug=True)
