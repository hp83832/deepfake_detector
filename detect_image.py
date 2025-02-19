import tkinter as tk
from tkinter import filedialog
from facenet_pytorch import MTCNN
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained deepfake detection model
model = tf.keras.models.load_model("models/deepfake_model.h5")

# Initialize MTCNN face detector
mtcnn = MTCNN(keep_all=True)

# Function to detect deepfake
def detect_deepfake(image_path):
    img = Image.open(image_path)
    faces, _ = mtcnn.detect(img)

    if faces is None:
        print("No face detected!")
        return

    for face in faces:
        x, y, w, h = [int(val) for val in face]
        cropped_face = img.crop((x, y, w, h))
        cropped_face = cropped_face.resize((128, 128))
        cropped_face = np.array(cropped_face) / 255.0
        cropped_face = np.expand_dims(cropped_face, axis=0)

        prediction = model.predict(cropped_face)[0][0]
        label = "Fake" if prediction > 0.5 else "Real"
        print(f"Prediction: {label}")

# Select image using file dialog
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        detect_deepfake(file_path)

# GUI Interface
root = tk.Tk()
tk.Button(root, text="Select Image", command=select_image).pack()
root.mainloop()