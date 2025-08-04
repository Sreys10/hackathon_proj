from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model("efficientnetb1_poultry_tuned.keras")

# Set class labels manually
class_names = ['Healthy', 'Coccidiosis', 'Newcastle Disease', 'Avian Influenza']  # Example

def prepare_image(file_path):
    img = image.load_img(file_path, target_size=(240, 240), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_data = prepare_image(filepath)
            prediction = model.predict(img_data)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            return render_template('index.html', prediction=predicted_class, confidence=confidence, image_url=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
