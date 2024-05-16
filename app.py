import os
import base64
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Define the path to your dataset directory
train_dir = r'archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'
valid_dir = r'archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid'
# Get the directory names as class names
class_names = sorted(os.listdir(train_dir))

# Define fertilizer mapping
fertilizer_mapping = {
    'Apple___Apple_scab': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Apple___Black_rot': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Apple___Cedar_apple_rust': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Apple___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Blueberry___healthy': 'Acidic Fertilizer for Acid-Loving Plants',
    'Cherry_(including_sour)_healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Cherry_(including_sour)_Powdery_mildew': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot': 'Nitrogen, Phosphorus, Potassium (NPK) Fertilizer',
    'Corn_(maize)Common_rust': 'Urea Fertilizer',
    'Corn_(maize)_Early_blight': 'Ammonium Nitrate Fertilizer',
    'Corn_(maize)_healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Corn_(maize)_Northern_Leaf_Blight': 'Potassium Sulfate Fertilizer',
    'Grape___Black_rot': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Grape__Esca(Black_Measles)': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Grape___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Orange__Haunglongbing(Citrus_greening)': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Peach___Bacterial_spot': 'Sulphate of Potash Fertilizer',
    'Peach___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Pepper,bell__Bacterial_spot': 'Potassium Nitrate Fertilizer',
    'Pepper,bell__healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Potato___Early_blight': 'Ammonium Phosphate Fertilizer',
    'Potato___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Potato___Late_blight': 'Potassium Chloride Fertilizer',
    'Potato___Leaf_mold': 'Urea Fertilizer',
    'Potato___Septoria_leaf_spot': 'Calcium Ammonium Nitrate Fertilizer',
    'Potato___Spider_mites': 'Monoammonium Phosphate Fertilizer',
    'Potato___Target_spot': 'Ammonium Sulfate Fertilizer',
    'Potato___Yellow_leaf_curl_virus': 'Potassium Sulfate Fertilizer',
    'Raspberry___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Soybean___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Squash___Powdery_mildew': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Strawberry___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Strawberry___Leaf_scorch': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Tomato___Bacterial_spot': 'Ammonium Nitrate Fertilizer',
    'Tomato___Early_blight': 'Triple Superphosphate Fertilizer',
    'Tomato___Late_blight': 'Calcium Nitrate Fertilizer',
    'Tomato___Leaf_mold': 'Potassium Sulfate Fertilizer',
    'Tomato___Septoria_leaf_spot': 'Potassium Nitrate Fertilizer',
    'Tomato___Spider_mites': 'Urea Fertilizer',
    'Tomato___Target_spot': 'Diammonium Phosphate Fertilizer',
    'Tomato___Yellow_leaf_curl_virus': 'Magnesium Sulfate Fertilizer',
}

# Function to preprocess the image
def preprocess_image(image):
    img = tf.image.resize(image, (128, 128))
    img = tf.expand_dims(img, axis=0)
    return img

# Function to make predictions
def predict_disease(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
    else:
        predicted_class = "Unknown"

    return predicted_class

# Function to get fertilizer recommendation
def get_fertilizer_recommendation(disease):
    return fertilizer_mapping.get(disease, "No specific recommendation found")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction_text='No file part')
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', prediction_text='No selected file')
        
        # If we have a file, save it
        if file:
            file_path = 'static/' + file.filename
            file.save(file_path)

            # Preprocess the image
            image = tf.keras.preprocessing.image.load_img(file_path)
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = predict_disease(file_path)
            
            # Get fertilizer recommendation
            recommendation = get_fertilizer_recommendation(prediction)

            # Encode image to base64
            with open(file_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

            return render_template('index.html', prediction_text=f'The plant disease is: {prediction}', fertilizer_recommendation=recommendation, image_data=encoded_image)

if __name__ == '__main__':
    app.run(debug=True)
