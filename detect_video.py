import cv2
import tensorflow as tf
import numpy as np
from facenet_pytorch import MTCNN
from tkinter import filedialog, Tk

# Load trained model
model = tf.keras.models.load_model("models/deepfake_model.h5")

# Initialize face detector
mtcnn = MTCNN(keep_all=True)

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces, _ = mtcnn.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = [int(val) for val in face]
                cropped_face = frame[y:h, x:w]
                cropped_face = cv2.resize(cropped_face, (128, 128)) / 255.0
                cropped_face = np.expand_dims(cropped_face, axis=0)

                prediction = model.predict(cropped_face)[0][0]
                label = "Fake" if prediction > 0.5 else "Real"
                color = (0, 0, 255) if label == "Fake" else (0, 255, 0)

                cv2.rectangle(frame, (x, y), (w, h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Deepfake Video Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Select video
root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
if video_path:
    detect_video(video_path)