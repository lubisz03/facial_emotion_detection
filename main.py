import cv2
from deepface import DeepFace

CONFIDENCE_THRESHOLD = 0.3

model = DeepFace.build_model("Emotion")

emotion_labels = ['angry', 'disgust', 'fear',
                  'happy', 'sad', 'surprise', 'neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Enable the default camera
cap = cv2.VideoCapture(0)

is_running = True
while is_running:
    # Read every frame from the cam
    ret, frame = cap.read()

    # If image not found shut down the program
    if not ret:
        print("Failed to grab frame")
        is_running = False

    # Transform image to grayscale and detect faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_roi_color = frame[y:y + h, x:x + w]

        # Process the detected image before making any predictions
        resized_face_color = cv2.resize(
            face_roi_color, (48, 48), interpolation=cv2.INTER_AREA)
        normalized_face_color = (resized_face_color / 255.0).astype('float32')
        reshaped_face_color = normalized_face_color.reshape(1, 48, 48, 3)

        # Make the face emotion prediction
        preds = model.predict(reshaped_face_color)

        # Check if the prediction's confidence is above a given treshold
        if (max(preds) < CONFIDENCE_THRESHOLD):
            continue

        # Set the detected facial emotion to be displayed
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # Put the detected emotion on the screen
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Facial Emotion Detection', frame)

    # Exit the program by clicking ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        is_running = False

# Shut down the window
cap.release()
cv2.destroyAllWindows()
