import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time

# --------------------------------
# Load Model
# --------------------------------
model = load_model("eye_model.h5")

# --------------------------------
# Haar Cascades
# --------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# --------------------------------
# Stability Buffer
# --------------------------------
history = deque(maxlen=20)

# --------------------------------
# Streamlit UI
# --------------------------------
st.set_page_config(
    page_title="Eye Detection AI",
    layout="centered"
)

st.title("👁 Real-Time Eye Detection AI")

st.write(
    "Live webcam-based eye state detection using TensorFlow + OpenCV"
)

# --------------------------------
# Video Processor
# --------------------------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.prev_time = time.time()

        self.frame_count = 0

        self.last_status = "Detecting..."
        self.last_color = (200, 200, 200)
        self.last_conf = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # --------------------------------
        # Resize for stability
        # --------------------------------
        img = cv2.resize(img, (640, 480))

        self.frame_count += 1

        # --------------------------------
        # FPS
        # --------------------------------
        now = time.time()

        fps = 1.0 / (now - self.prev_time + 1e-6)

        self.prev_time = now

        # --------------------------------
        # Process every 2nd frame
        # --------------------------------
        process_this_frame = self.frame_count % 2 == 0

        if process_this_frame:

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            status = "No Eyes"
            color = (0, 165, 255)
            smooth_conf = 0

            for (x, y, w, h) in faces:

                face_gray = gray[y:y+h, x:x+w]

                face_color = img[y:y+h, x:x+w]

                # Brightness enhancement
                face_color = cv2.convertScaleAbs(
                    face_color,
                    alpha=1.1,
                    beta=10
                )

                eyes = eye_cascade.detectMultiScale(
                    face_gray,
                    scaleFactor=1.1,
                    minNeighbors=7
                )

                eyes = sorted(
                    eyes,
                    key=lambda e: e[2] * e[3],
                    reverse=True
                )[:2]

                eye_states = []

                for (ex, ey, ew, eh) in eyes:

                    eye = face_color[
                        ey + int(0.2 * eh):
                        ey + int(0.8 * eh),

                        ex + int(0.1 * ew):
                        ex + int(0.9 * ew)
                    ]

                    if eye.size == 0:
                        continue

                    # Blur noise
                    eye = cv2.GaussianBlur(
                        eye,
                        (3, 3),
                        0
                    )

                    # Resize
                    eye = cv2.resize(
                        eye,
                        (64, 64)
                    )

                    # Normalize
                    eye = eye / 255.0

                    # Reshape
                    eye = eye.reshape(
                        1,
                        64,
                        64,
                        3
                    )

                    # Prediction
                    pred = model.predict(
                        eye,
                        verbose=0
                    )[0][0]

                    if pred > 0.65:
                        eye_states.append(1)

                    elif pred < 0.35:
                        eye_states.append(0)

                # --------------------------------
                # Stable Prediction
                # --------------------------------
                if len(eye_states) > 0:

                    avg = sum(eye_states) / len(eye_states)

                    history.append(avg)

                    smooth_conf = (
                        sum(history) /
                        len(history)
                    )

                    if (
                        len(history) >= 12
                        and smooth_conf < 0.45
                    ):

                        status = "Sleepy 😴"

                        color = (0, 80, 255)

                    else:

                        status = "Awake 👀"

                        color = (0, 220, 0)

            # Cache results
            self.last_status = status
            self.last_color = color
            self.last_conf = smooth_conf

        # --------------------------------
        # Use cached results
        # --------------------------------
        status = self.last_status
        color = self.last_color
        smooth_conf = self.last_conf

        # --------------------------------
        # Top Header
        # --------------------------------
        cv2.rectangle(
            img,
            (0, 0),
            (img.shape[1], 60),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            img,
            "Real-Time Eye Detection AI",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # FPS
        cv2.putText(
            img,
            f"FPS: {fps:.1f}",
            (img.shape[1] - 160, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2
        )

        # --------------------------------
        # Status
        # --------------------------------
        cv2.putText(
            img,
            f"Status: {status}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # --------------------------------
        # Confidence Bar
        # --------------------------------
        bar_x = 20
        bar_y = 125
        bar_width = 250
        bar_height = 18

        cv2.rectangle(
            img,
            (bar_x, bar_y),
            (bar_x + bar_width,
             bar_y + bar_height),
            (60, 60, 60),
            -1
        )

        fill_width = int(
            bar_width * smooth_conf
        )

        cv2.rectangle(
            img,
            (bar_x, bar_y),
            (bar_x + fill_width,
             bar_y + bar_height),
            color,
            -1
        )

        cv2.putText(
            img,
            f"Confidence: {smooth_conf:.2f}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# --------------------------------
# Webcam Stream
# --------------------------------
webrtc_streamer(
    key="eye-detection",

    video_processor_factory=VideoProcessor,

    media_stream_constraints={
        "video": {
            "width": 480,
            "height": 360
        },
        "audio": False
    },

    video_html_attrs={
        "autoPlay": True,
        "controls": False,
        "muted": True,
    },

    async_processing=True
)