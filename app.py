
from pathlib import Path
from collections import Counter, deque
from threading import Lock

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# Text-to-speech is optional because it may not work on Streamlit Cloud.
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SignSpeak | AI Sign Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "asl_model.pkl"

HISTORY_SIZE = 15
STABLE_COUNT = 10


# ============================================================
# 2. CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --background: #101020;
        --surface: #1b1b32;
        --purple: #a78bfa;
        --pink: #f472b6;
        --cyan: #67e8f9;
        --text: #f8f7ff;
        --muted: #aaa9c4;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 5% 0%,
            rgba(124, 58, 237, 0.23), transparent 38%),
          radial-gradient(ellipse at 95% 10%,
            rgba(236, 72, 153, 0.15), transparent 32%),
          linear-gradient(135deg, #101020 0%, #17152c 55%,
            #101a2b 100%);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        font-family: 'DM Sans', sans-serif;
    }

    h1 {
        color: #f8f7ff;
        font-weight: 800;
    }

    .hero {
        padding: 1.5rem 0 1rem 0;
    }

    .hero-label {
        color: #c4b5fd;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        line-height: 1.1;
        font-weight: 800;
        margin: 0.5rem 0;
        color: #ffffff;
    }

    .hero-title span {
        background: linear-gradient(90deg, #a78bfa, #f472b6, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        color: #c2c0dc;
        font-size: 1.05rem;
        max-width: 750px;
        line-height: 1.7;
    }

    .panel {
        background: rgba(27, 27, 50, 0.82);
        border: 1px solid rgba(167, 139, 250, 0.22);
        border-radius: 20px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .panel-label {
        color: #aaa9c4;
        text-transform: uppercase;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    .prediction {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1.2;
        text-align: center;
        color: #c4b5fd;
    }

    .muted {
        color: #aaa9c4;
    }

    .sentence {
        min-height: 85px;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(103, 232, 249, 0.25);
        background: rgba(16, 16, 32, 0.7);
        color: #f8f7ff;
        font-size: 1.25rem;
        overflow-wrap: anywhere;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.8rem;
    }

    [data-testid="stMetric"] {
        background: rgba(27, 27, 50, 0.75);
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(167, 139, 250, 0.18);
    }

    hr {
        border-color: rgba(167, 139, 250, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD THE TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Make sure models/asl_model.pkl is included in your repository."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)


# ============================================================
# 4. WEBCAM PROCESSOR AND SIGN RECOGNITION
# ============================================================

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.history = deque(maxlen=HISTORY_SIZE)
        self.accepted_letters = deque()
        self.lock = Lock()

        self.latest_letter = ""
        self.latest_confidence = 0.0
        self.hand_detected = False
        self.last_added_letter = None
        self.processing_error = ""

    def recv(self, frame):
        try:
            image = frame.to_ndarray(format="bgr24")
            image = cv2.flip(image, 1)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            detected = bool(results.multi_hand_landmarks)

            with self.lock:
                self.hand_detected = detected
                self.processing_error = ""

            if detected:
                hand = results.multi_hand_landmarks[0]

                self.mp_drawing.draw_landmarks(
                    image,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS,
                )

                features = []
                for point in hand.landmark:
                    features.extend([point.x, point.y, point.z])

                # The model expects 63 values per hand.
                X = pd.DataFrame([features])

                prediction = str(self.model.predict(X)[0])
                confidence = 0.0

                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(X)[0]
                    confidence = float(np.max(probabilities))

                with self.lock:
                    self.latest_letter = prediction
                    self.latest_confidence = confidence

                self.history.append(prediction)

                # Accept a letter only when it is stable across frames.
                if len(self.history) == HISTORY_SIZE:
                    letter, occurrences = Counter(
                        self.history
                    ).most_common(1)[0]

                    if (
                        occurrences >= STABLE_COUNT
                        and letter != self.last_added_letter
                    ):
                        with self.lock:
                            self.accepted_letters.append(letter)

                        self.last_added_letter = letter
                        self.history.clear()

                cv2.putText(
                    image,
                    f"Prediction: {prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (190, 150, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    image,
                    f"Confidence: {confidence:.0%}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 240, 240),
                    2,
                    cv2.LINE_AA,
                )

            else:
                self.history.clear()
                self.last_added_letter = None

                with self.lock:
                    self.latest_letter = ""
                    self.latest_confidence = 0.0

                cv2.putText(
                    image,
                    "Show your hand to the camera",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24",
            )

        except Exception as exc:
            # Keep the video processor alive and expose the error in the UI.
            with self.lock:
                self.processing_error = str(exc)

            try:
                image = frame.to_ndarray(format="bgr24")
                cv2.putText(
                    image,
                    "Processing error - check the app status",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                return av.VideoFrame.from_ndarray(
                    image,
                    format="bgr24",
                )
            except Exception:
                return frame


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">AI-powered communication</div>
        <div class="hero-title">Sign<span>Speak</span> 🤟</div>
        <div class="hero-description">
            Transform American Sign Language alphabet gestures into text
            with real-time hand tracking and machine learning.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

if model_error:
    st.error(f"Could not load the ASL model: {model_error}")
    st.info(
        "Check that models/asl_model.pkl exists and that the required "
        "Python packages are installed."
    )
    st.stop()


# ============================================================
# 6. SESSION STATE
# ============================================================

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "camera_started" not in st.session_state:
    st.session_state.camera_started = False


# ============================================================
# 7. MAIN LAYOUT
# ============================================================

camera_col, result_col = st.columns([1.35, 1], gap="large")

with camera_col:
    st.markdown("### 📷 Live camera")
    st.caption("Allow camera access when your browser asks for permission.")

    ctx = webrtc_streamer(
        key="signspeak-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
    )

    st.caption(
        "Tip: keep one hand visible, use good lighting, and hold each "
        "letter steady briefly."
    )

with result_col:
    st.markdown("### ✨ Recognition")

    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    status_placeholder = st.empty()
    error_placeholder = st.empty()


# ============================================================
# 8. LIVE RECOGNITION STATUS
# ============================================================

@st.fragment(run_every="500ms")
def update_live_status():
    processor = ctx.video_processor

    if not ctx.state.playing:
        prediction_placeholder.markdown(
            """
            <div class="panel">
                <div class="panel-label">Current letter</div>
                <div class="prediction">—</div>
                <p class="muted" style="text-align:center">
                    Start the camera to begin recognition.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        confidence_placeholder.metric("Confidence", "—")
        status_placeholder.info("Camera is not running.")
        return

    if processor is None:
        status_placeholder.warning(
            "Connecting to the camera and initializing the model..."
        )
        return

    with processor.lock:
        letter = processor.latest_letter
        confidence = processor.latest_confidence
        detected = processor.hand_detected
        error = processor.processing_error

    shown_letter = letter if letter else "—"

    prediction_placeholder.markdown(
        f"""
        <div class="panel">
            <div class="panel-label">Current letter</div>
            <div class="prediction">{shown_letter}</div>
            <p class="muted" style="text-align:center">
                {"Hand detected" if detected else "Waiting for a hand"}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confidence_placeholder.metric(
        "Confidence",
        f"{confidence:.1%}" if letter else "—",
    )

    if error:
        error_placeholder.error(f"Prediction error: {error}")
    else:
        error_placeholder.empty()

    if detected:
        status_placeholder.success(
            "Hand detected. Hold your sign steady to add a letter."
        )
    else:
        status_placeholder.info(
            "Show your hand clearly in front of the camera."
        )


update_live_status()


# ============================================================
# 9. BUILD WORDS AND SENTENCES
# ============================================================

st.divider()
st.markdown("### 📝 Your message")
st.write(
    "Recognized letters are added automatically when the same sign "
    "remains stable. You can also edit the text yourself."
)

processor = ctx.video_processor

# Transfer recognized letters from the video thread into Streamlit state.
if processor is not None:
    with processor.lock:
        while processor.accepted_letters:
            new_letter = processor.accepted_letters.popleft()
            st.session_state.sentence += new_letter

st.text_area(
    "Recognized text",
    key="sentence",
    height=130,
    placeholder="Your recognized letters will appear here...",
)

button_col1, button_col2, button_col3, button_col4 = st.columns(4)

with button_col1:
    if st.button("␣ Add space", use_container_width=True):
        st.session_state.sentence += " "
        st.rerun()

with button_col2:
    if st.button("⌫ Delete last", use_container_width=True):
        st.session_state.sentence = st.session_state.sentence[:-1]
        st.rerun()

with button_col3:
    if st.button("🧹 Clear text", use_container_width=True):
        st.session_state.sentence = ""
        st.rerun()

with button_col4:
    if st.button("➕ Add punctuation", use_container_width=True):
        st.session_state.sentence += "."
        st.rerun()


# ============================================================
# 10. TEXT-TO-SPEECH
# ============================================================

st.divider()
st.markdown("### 🔊 Text to speech")

st.write(
    "Read your message aloud using the text-to-speech engine, if "
    "the environment supports audio output."
)

if st.button("🔊 Speak message", type="primary"):
    message = st.session_state.sentence.strip()

    if not message:
        st.warning("Recognize some letters or type a message first.")

    elif pyttsx3 is None:
        st.warning(
            "The pyttsx3 package is not installed in this environment."
        )

    else:
        try:
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
            engine.stop()

            st.success("Speech playback was requested.")

        except Exception as exc:
            st.error(
                "Text-to-speech could not run in this environment. "
                f"Details: {exc}"
            )
            st.info(
                "On Streamlit Cloud, pyttsx3 may not play sound on your "
                "personal device because it runs on the server."
            )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#aaa9c4; padding:1rem 0;">
        <p><strong>SignSpeak</strong> · AI-powered ASL alphabet recognition</p>
        <p style="font-size:0.85rem;">
            Built with Python, Streamlit, MediaPipe and Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
