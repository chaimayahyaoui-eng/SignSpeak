
from pathlib import Path
from collections import Counter, deque
from threading import Lock

import av
import cv2
import joblib
import mediapipe as mp
import pandas as pd
import pyttsx3
import streamlit as st
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

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


# ==================================================
# 2. CUSTOM COLORFUL DESIGN
# ==================================================

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
        font-family: 'DM Sans', sans-serif !important;
    }

    h1 {
        color: #fff !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px;
    }

    h2, h3 {
        color: #f7f4ff !important;
        font-weight: 700 !important;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .hero {
        padding: 28px 30px;
        border-radius: 26px;
        margin-bottom: 24px;
        background: linear-gradient(
            115deg,
            rgba(124, 58, 237, 0.33),
            rgba(219, 39, 119, 0.22),
            rgba(8, 145, 178, 0.18)
        );
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 16px 50px rgba(0,0,0,0.15);
    }

    .brand {
        font-size: 14px;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 800;
        color: #d8ccff;
    }

    .hero-title {
        font-size: clamp(32px, 4vw, 48px);
        font-weight: 800;
        line-height: 1.15;
        margin: 10px 0;
        color: white;
    }

    .hero-description {
        font-size: 15px;
        line-height: 1.8;
        color: #d4d2e9;
        max-width: 700px;
    }

    .panel {
        background: linear-gradient(
            145deg,
            rgba(35, 33, 61, 0.96),
            rgba(24, 27, 47, 0.96)
        );
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.14);
    }

    .section-label {
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 800;
        text-transform: uppercase;
        color: #b9a5ff;
        margin-bottom: 7px;
    }

    .panel-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .panel-description {
        color: #aaa9c4;
        font-size: 13px;
        line-height: 1.7;
    }

    .letter-card {
        border-radius: 20px;
        padding: 18px;
        text-align: center;
        background: linear-gradient(
            135deg,
            rgba(139, 92, 246, 0.24),
            rgba(236, 72, 153, 0.16),
            rgba(34, 211, 238, 0.12)
        );
        border: 1px solid rgba(196,181,253,0.25);
        margin: 12px 0;
    }

    .letter-label {
        color: #c4b5fd;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .letter-value {
        color: #ffffff;
        font-size: 64px;
        line-height: 1.2;
        font-weight: 800;
        padding: 8px 0;
    }

    .message-card {
        min-height: 140px;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
        border-radius: 17px;
        padding: 20px;
        color: #ffffff;
        font-size: 20px;
        line-height: 1.8;
        background: rgba(12, 14, 30, 0.75);
        border: 1px solid rgba(167,139,250,0.25);
    }

    .empty-message {
        color: #85849f;
        font-size: 15px;
    }

    .status {
        display: inline-block;
        border-radius: 30px;
        padding: 7px 12px;
        background: rgba(34,197,94,0.12);
        color: #86efac;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid rgba(34,197,94,0.2);
    }

    .status-idle {
        background: rgba(167,139,250,0.12);
        color: #d8ccff;
        border-color: rgba(167,139,250,0.22);
    }

    .stButton > button {
        min-height: 43px;
        width: 100%;
        border-radius: 13px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.12);
        color: #f8f7ff;
        background: rgba(255,255,255,0.07);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        color: #ffffff;
        border-color: #a78bfa;
        background: rgba(139,92,246,0.23);
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #7c3aed, #db2777);
        border: none;
        color: white;
    }

    [data-testid="stWebRtcStreamer"] {
        border-radius: 16px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    .footer {
        text-align: center;
        color: #85849f;
        font-size: 12px;
        padding-top: 24px;
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem;
        }

        .hero {
            padding: 22px;
        }

        .panel {
            padding: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ==================================================
# 4. CAMERA PROCESSOR
# ==================================================

class SignLanguageProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = load_model()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
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

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        detected = bool(results.multi_hand_landmarks)

        with self.lock:
            self.hand_detected = detected

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

            try:
                X = pd.DataFrame([features])
                prediction = str(self.model.predict(X)[0])

                confidence = 0.0

                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(X)[0]
                    confidence = float(probabilities.max())

                with self.lock:
                    self.latest_letter = prediction
                    self.latest_confidence = confidence

                self.history.append(prediction)

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
                    f"Letter: {prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (170, 110, 255),
                    2,
                )

                cv2.putText(
                    image,
                    f"Confidence: {confidence * 100:.1f}%",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (90, 230, 255),
                    2,
                )

            except Exception as exc:
                print("Prediction error:", exc)

                cv2.putText(
                    image,
                    "Prediction error",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        else:
            self.history.clear()
            self.last_added_letter = None

            with self.lock:
                self.latest_letter = ""
                self.latest_confidence = 0.0

            cv2.putText(
                image,
                "Show your hand",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 100),
                2,
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ==================================================
# 5. SESSION STATE
# ==================================================

if "message" not in st.session_state:
    st.session_state.message = ""

if "speech_notice" not in st.session_state:
    st.session_state.speech_notice = ""


# ==================================================
# 6. HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="brand">🤟 SIGNSPEAK · AI STUDIO</div>
        <div class="hero-title">
            Your hands.<br>
            Your voice. Your world.
        </div>
        <div class="hero-description">
            Transform ASL alphabet gestures into written messages
            and spoken words with real-time computer vision.
            Sign naturally, build your message, and let it speak.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

try:
    load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


# ==================================================
# 7. MAIN TWO-COLUMN LAYOUT
# ==================================================

camera_col, workspace_col = st.columns(
    [1.12, 0.88],
    gap="large",
)

with camera_col:
    st.markdown(
        """
        <div class="panel">
            <div class="section-label">01 / LIVE STUDIO</div>
            <div class="panel-title">Gesture recognition</div>
            <div class="panel-description">
                Position your hand inside the camera.
                Hold each sign steady until it is recognized.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctx = webrtc_streamer(
        key="signspeak-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown(
        """
        <div class="panel">
            <div class="section-label">HOW TO USE</div>
            <div class="panel-description">
                <b>1.</b> Show an ASL alphabet sign.<br>
                <b>2.</b> Hold it steady for a moment.<br>
                <b>3.</b> Remove your hand before repeating
                the same letter.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with workspace_col:
    st.markdown(
        """
        <div class="panel">
            <div class="section-label">02 / RECOGNITION</div>
            <div class="panel-title">Your sign</div>
            <div class="panel-description">
                The AI's latest prediction appears below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    letter_display = st.empty()
    status_display = st.empty()
    confidence_display = st.empty()

    st.markdown(
        """
        <div class="panel">
            <div class="section-label">03 / MESSAGE BUILDER</div>
            <div class="panel-title">Make it meaningful</div>
            <div class="panel-description">
                Your recognized letters are collected here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    message_display = st.empty()
    count_display = st.empty()

    st.markdown(
        """
        <div class="panel">
            <div class="section-label">04 / CONTROLS</div>
            <div class="panel-title">Edit & communicate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control1, control2 = st.columns(2)

    with control1:
        if st.button(
            "＋  Add space",
            use_container_width=True,
        ):
            st.session_state.message += " "
            st.rerun()

        if st.button(
            "⌫  Delete letter",
            use_container_width=True,
        ):
            st.session_state.message = (
                st.session_state.message[:-1]
            )
            st.rerun()

    with control2:
        if st.button(
            "✦  Speak message",
            type="primary",
            use_container_width=True,
        ):
            text_to_speak = (
                st.session_state.message.strip()
            )

            if not text_to_speak:
                st.session_state.speech_notice = (
                    "Add some letters before speaking."
                )
            else:
                try:
                    engine = pyttsx3.init()
                    engine.say(text_to_speak)
                    engine.runAndWait()
                    engine.stop()

                    st.session_state.speech_notice = (
                        "Speech playback finished."
                    )
                except Exception as exc:
                    st.session_state.speech_notice = (
                        f"Speech failed: {exc}"
                    )

            st.rerun()

        if st.button(
            "↺  Clear message",
            use_container_width=True,
        ):
            st.session_state.message = ""
            st.session_state.speech_notice = ""
            st.rerun()

    if st.session_state.speech_notice:
        st.info(st.session_state.speech_notice)


# ==================================================
# 8. LIVE UI REFRESH
# ==================================================

@st.fragment(run_every="300ms")
def update_live_interface():
    processor = ctx.video_processor

    if processor is not None:
        with processor.lock:
            while processor.accepted_letters:
                letter = processor.accepted_letters.popleft()
                st.session_state.message += letter

            letter = processor.latest_letter
            confidence = processor.latest_confidence
            hand_detected = processor.hand_detected

        if hand_detected:
            letter_display.markdown(
                f"""
                <div class="letter-card">
                    <div class="letter-label">
                        CURRENT PREDICTION
                    </div>
                    <div class="letter-value">
                        {letter if letter else "…"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            status_display.markdown(
                '<span class="status">● HAND DETECTED</span>',
                unsafe_allow_html=True,
            )

            confidence_display.progress(
                max(0.0, min(1.0, confidence)),
                text=f"Model confidence: {confidence * 100:.1f}%",
            )
        else:
            letter_display.markdown(
                """
                <div class="letter-card">
                    <div class="letter-label">
                        CURRENT PREDICTION
                    </div>
                    <div class="letter-value">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            status_display.markdown(
                '<span class="status status-idle">'
                '○ SHOW YOUR HAND</span>',
                unsafe_allow_html=True,
            )

            confidence_display.empty()

    else:
        letter_display.markdown(
            """
            <div class="letter-card">
                <div class="letter-label">CURRENT PREDICTION</div>
                <div class="letter-value">—</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        status_display.markdown(
            '<span class="status status-idle">'
            '○ CAMERA NOT STARTED</span>',
            unsafe_allow_html=True,
        )

        confidence_display.empty()

    message = st.session_state.message

    if message:
        # Escape user text before inserting it into HTML.
        import html
        safe_message = html.escape(message)

        message_display.markdown(
            f'<div class="message-card">{safe_message}</div>',
            unsafe_allow_html=True,
        )
    else:
        message_display.markdown(
            """
            <div class="message-card">
                <span class="empty-message">
                    Your message will appear here as you sign...
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    count_display.caption(
        f"{len(message)} characters · "
        f"{len(message.split())} words"
    )


update_live_interface()


# ==================================================
# 9. FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        SIGNSPEAK · Powered by MediaPipe, OpenCV & Machine Learning
        <br><br>
        Current version recognizes ASL alphabet signs and spells
        messages letter by letter. Full ASL word and grammar
        translation is not yet supported.
    </div>
    """,
    unsafe_allow_html=True,
)
