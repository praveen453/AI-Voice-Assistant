# drowsy_system_with_voice.py

import os
import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import speech_recognition as sr

from voice_assistant import AIVoiceAssistant
from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# TensorFlow log level
# -----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------
# Model & detection settings
# -----------------------------
MODEL_PATH = "Models/drowsiness_model_mobilenet.h5"
LABELS = ["Drowsy", "Non-Drowsy"]
IMG_SIZE = (224, 224)

EYE_CLOSED_FRAMES_THRESH = 10   # consecutive closed-eye frames
WARMUP_DURATION = 5.0           # seconds before any alert
ALERT_COOLDOWN = 3.0            # seconds between alerts

# MIC index (use same index from mic_test.py)
MIC_INDEX = 5    # change if your working mic index is different

# -----------------------------
# Global state
# -----------------------------
closed_frames = 0
invert_logic = False
start_time = time.time()
alert_active = False
last_alert_time = 0.0

conversation_mode = False       # are we in continuous talk mode?
conversation_thread = None

# -----------------------------
# Voice assistant (Gemini + offline fallback)
# -----------------------------
assistant = AIVoiceAssistant(
    driver_name="Praveen",
    use_cloud_assistant=True,           # use Gemini if API key ok
    gemini_model_name="gemini-2.5-flash"
)

# -----------------------------
# Context helpers (for proposal)
# -----------------------------
def get_speed() -> float:
    return 60.0

def get_weather() -> str:
    return "clear"

# -----------------------------
# One-shot drowsy alert (no blocking)
# -----------------------------
def trigger_voice_alert(level: str | None, custom_message: str | None = None):
    """
    Non-blocking wrapper: speak a drowsiness alert in a background thread.
    level: "low"/"medium"/"high" → uses assistant.alert_drowsiness
    custom_message: direct text
    """
    global alert_active, last_alert_time

    now = time.time()

    # warm-up period
    if now - start_time < WARMUP_DURATION:
        return

    # anti-spam
    if now - last_alert_time < ALERT_COOLDOWN:
        return

    last_alert_time = now

    def _run():
        global alert_active
        if alert_active:
            return
        alert_active = True
        try:
            context = {
                "speed": get_speed(),
                "weather": get_weather(),
                "location_hint": "on this route",
            }

            if custom_message and level is None:
                assistant.speak(custom_message)
            elif level is not None:
                assistant.alert_drowsiness(level, context)
            else:
                assistant.speak("Please stay focused on the road.")
        finally:
            alert_active = False

    threading.Thread(target=_run, daemon=True).start()

# ==========================================================
#  Continuous conversation loop (runs in background thread)
# ==========================================================
def listen_once(recognizer: sr.Recognizer, microphone: sr.Microphone) -> str:
    """Listen once from mic and return recognized text."""
    with microphone as source:
        print("[System] Listening to driver...")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        audio = recognizer.listen(source, phrase_time_limit=5)

    try:
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"[Driver] {text}")
        return text
    except sr.UnknownValueError:
        print("[System] Could not understand driver.")
        return ""
    except sr.RequestError as e:
        print(f"[System] Speech service error: {e}")
        return ""

def conversation_loop():
    """
    Runs while conversation_mode == True.
    Uses Gemini (via assistant.handle_command_with_nlp_backend)
    until driver says 'exit assistant' / 'stop talking'.
    """
    global conversation_mode

    recognizer = sr.Recognizer()

    try:
        microphone = sr.Microphone(device_index=MIC_INDEX)
    except Exception as e:
        print(f"[System] Could not open mic {MIC_INDEX}, using default. Error: {e}")
        microphone = sr.Microphone()

    # Initial instruction to driver
    #assistant.speak(
    #    "Praveen, I will stay with you and keep talking while you are driving. "
    #    "You can ask me questions. If you want me to stop talking, say 'exit assistant'."
   # )

    while conversation_mode:
        text = listen_once(recognizer, microphone)

        if not text:
            # just continue the loop without crashing
            continue

        low = text.lower().strip()

        # driver manually ends conversation
        if ("exit assistant" in low) or (low == "exit") or ("stop talking" in low):
            conversation_mode = False
            assistant.speak(
                "Okay, I will stop talking now, but I will keep monitoring your drowsiness."
            )
            break

        # send to AI assistant (Gemini + offline rules)
        try:
            assistant.handle_command_with_nlp_backend(text)
        except SystemExit:
            # in case offline handler raises SystemExit for 'exit'
            conversation_mode = False
            break
        except Exception as e:
            print(f"[System] Error in conversation loop: {e}")

    print("[System] Conversation mode ended.")

def start_conversation_mode():
    """
    Start continuous Q&A mode only once (when first drowsy detected).
    """
    global conversation_mode, conversation_thread

    if conversation_mode:
        return

    conversation_mode = True
    conversation_thread = threading.Thread(target=conversation_loop, daemon=True)
    conversation_thread.start()

# -----------------------------
# Main application
# -----------------------------
def main():
    global closed_frames, invert_logic, start_time

    # 1. Load CNN model
    if not os.path.exists(MODEL_PATH):
        print(f"[System] Error: Model file not found at {MODEL_PATH}")
        return

    print("[System] Loading CNN drowsiness model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[System] ✅ Model loaded successfully.")
    except Exception as e:
        print(f"[System] Failed to load model: {e}")
        return

    start_time = time.time()
    closed_frames = 0

    # 2. Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[System] Error: Could not open webcam.")
        return

    assistant.speak(
        "Drowsiness detection assistant started. "
        "I will warn you if I detect signs of sleepiness."
    )

    # 3. Haar cascades
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    print("--- Drowsiness Detection + AI Voice Assistant ---")
    print("Press 'q' to Quit | 'i' to Invert CNN logic")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[System] Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(100, 100)
        )

        cnn_status = "Unknown"
        is_cnn_drowsy = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # --- 1) Eye detection ---
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, 1.1, 4, minSize=(20, 20)
            )

            if len(eyes) == 0:
                closed_frames += 1
            else:
                closed_frames = 0

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (0, 255, 0),
                    1,
                )

            # --- 2) CNN inference ---
            try:
                face_img = frame[y:y + h, x:x + w]
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(face_img_rgb, IMG_SIZE)
                normalized = tf.keras.applications.mobilenet_v2.preprocess_input(
                    resized.astype("float32")
                )
                reshaped = np.reshape(normalized, (1, 224, 224, 3))

                prediction = model.predict(reshaped, verbose=0)
                raw_drowsy_score = float(prediction[0][0])
                raw_non_drowsy_score = float(prediction[0][1])

                is_cnn_drowsy = raw_drowsy_score > raw_non_drowsy_score
                if invert_logic:
                    is_cnn_drowsy = not is_cnn_drowsy

                if is_cnn_drowsy:
                    cnn_status = f"DROWSY ({raw_drowsy_score:.2f})"
                else:
                    cnn_status = f"Active ({raw_non_drowsy_score:.2f})"
            except Exception as e:
                print(f"[System] CNN Error: {e}")
                cnn_status = "Error"
                is_cnn_drowsy = False

            # --- 3) Final status & voice behaviour ---
            final_status = "Scanning..."
            alert_color = (0, 255, 0)

            # A) Eyes closed → strong drowsy
            if closed_frames > EYE_CLOSED_FRAMES_THRESH:
                final_status = "EYES CLOSED!"
                alert_color = (0, 0, 255)

                trigger_voice_alert("high")
                start_conversation_mode()   # start Q&A mode

            # B) CNN says drowsy
            elif is_cnn_drowsy:
                final_status = f"CNN: {cnn_status}"
                alert_color = (0, 0, 255)

                trigger_voice_alert("medium")
                start_conversation_mode()

            # C) Active (no extra voice here; conversation mode continues
            #    separately until driver says 'exit assistant')
            else:
                final_status = "Active"
                alert_color = (0, 255, 0)

            # overlays
            cv2.putText(
                frame,
                final_status,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                alert_color,
                2,
            )

            cv2.putText(
                frame,
                f"Closed Frames: {closed_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"Model: {cnn_status}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )

            current_time = time.time()
            if current_time - start_time < WARMUP_DURATION:
                remaining = int(WARMUP_DURATION - (current_time - start_time))
                cv2.putText(
                    frame,
                    f"WARMUP: {remaining}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Drowsiness Detector + AI Voice Assistant", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("i"):
            invert_logic = not invert_logic
            print(f"[System] Logic Inverted: {invert_logic}")

    cap.release()
    cv2.destroyAllWindows()
    print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()
