from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'diseases_deteection.keras'
CLASS_NAMES = ['cocci', 'salmo', 'healthy', 'ncd']  # Your 4 classes

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    input_shape = model.input_shape[1:3]  # Get (height, width)
    print(f"Model loaded. Input shape: {input_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    input_shape = (224, 224)  # Default size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        img = image.load_img(img_path, target_size=input_shape, color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array)
        predicted_idx = np.argmax(preds[0])
        
        # Ensure prediction is within our class range
        if predicted_idx >= len(CLASS_NAMES):
            raise ValueError(f"Model predicted invalid class index: {predicted_idx}")
            
        return CLASS_NAMES[predicted_idx], round(100 * np.max(preds[0]), 2)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    img_url = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_url = url_for('static', filename=f'uploads/{filename}')
            
            try:
                class_name, confidence = predict_image(filepath)
                prediction = {
                    'class': class_name,
                    'confidence': confidence,
                    'img_url': img_url
                }
            except Exception as e:
                prediction = {'error': str(e)}
    
    return render_template('index.html', 
                         prediction=prediction,
                         model_loaded=model is not None,
                         class_names=CLASS_NAMES)

if __name__ == '__main__':
    app.run(debug=True)