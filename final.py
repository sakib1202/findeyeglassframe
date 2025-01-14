import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2 as cv

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CATEGORIES = ["heart", "long", "oval", "round", "square"]
RECOMMENDATIONS = {
    "heart": "Round or oval frames",
    "long": "Tall frames with decorative temples",
    "oval": "Square or rectangular frames",
    "round": "Square or angular frames",
    "square": "Round or oval frames"
}

# Function to build the model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CATEGORIES), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to predict face shape and recommend glasses
def recommend_glasses(model, image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_shape = CATEGORIES[predicted_index]
    return predicted_shape, RECOMMENDATIONS.get(predicted_shape, "No recommendation available")

# Streamlit UI
st.title("Face Shape Detector and Glasses Recommendation")
st.write("Upload a face image to detect its shape and get personalized glasses recommendations.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load and Train Model
st.write("Training the model...")
model = build_model()

# Placeholder for dataset paths (replace with actual paths in deployment)
TRAIN_DATASET = "/content/drive/MyDrive/Face set/Train face"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    TRAIN_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    TRAIN_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

model.fit(train_generator, epochs=5, validation_data=validation_generator)
st.write("Model trained successfully!")

# Prediction
if uploaded_file is not None:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(file_path, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing image...")
    predicted_shape, frame_recommendation = recommend_glasses(model, file_path)

    st.write(f"**Predicted Face Shape:** {predicted_shape}")
    st.write(f"**Recommended Glasses Frame:** {frame_recommendation}")

    # Cleanup
    os.remove(file_path)
