import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# --------------------------
# Load the pre-trained emotion detection model
# --------------------------
model_path = 'emotion_model.hdf5'  # Path where you saved the model file
emotion_classifier = load_model(model_path, compile=False)

# List of emotion labels – update if your model uses different labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# --------------------------
# Load OpenCV's Haar cascade for face detection
# --------------------------
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --------------------------
# Start video capture
# --------------------------
camera = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale (face detection generally works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (face) from the grayscale image
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))  # Resize to the size expected by the model

        # Preprocess the ROI: convert to array and scale pixel values
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict the emotion on the ROI
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Draw a rectangle around the face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{label}: {emotion_probability*100:.2f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-time Face Emotion Detection", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
