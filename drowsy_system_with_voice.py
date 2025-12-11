# drowsy_system_with_voice.py

import os
import cv2
import numpy as np
import tensorflow as tf
import threading
import time

from voice_assistant import AIVoiceAssistant  # <- oyāge AI voice module eka

# --- Mute TensorFlow Logs ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Configuration (from drowsiness_app) ---  :contentReference[oaicite:0]{index=0}
MODEL_PATH = "Models/drowsiness_model_mobilenet.h5"
LABELS = ["Drowsy", "Non-Drowsy"]
IMG_SIZE = (224, 224)

# Thresholds & Parameters
EYE_CLOSED_FRAMES_THRESH = 10   # Consecutive frames with 0 eyes to trigger alert
WARMUP_DURATION = 5.0           # Seconds to wait before alerting
ALERT_COOLDOWN = 3.0            # Seconds between alerts

# --- State Management ---
alert_active = False
last_alert_time = 0.0
start_time = time.time()
closed_frames = 0
invert_logic = False

# -------------------------------------------------
# 1. Create AI Voice Assistant (offline drowsy alerts)
# ------------------------------------------------- :contentReference[oaicite:1]{index=1}
assistant = AIVoiceAssistant(
    driver_name="Praveen",
    use_cloud_assistant=True,   # auto drowsy alerts use OFFLINE TTS
    gemini_model_name="gemini-2.5-flash" # gemini_model_name can stay default
)

# Extra context helpers (for proposal: speed + weather + time)
def get_speed() -> float:
    """TODO: Replace with real OBD/GPS speed if available."""
    return 60.0

def get_weather() -> str:
    """TODO: Connect to real weather API if needed."""
    return "clear"


# -------------------------------------------------
# 2. Voice alert wrapper (integrates detection -> assistant)
# -------------------------------------------------
def trigger_voice_alert(level: str | None, custom_message: str | None = None):
    """
    Wrapper around AIVoiceAssistant for safe, non-spammy alerts.
    - level: "low", "medium", "high"  (if None, we only speak custom_message)
    - custom_message: direct text to speak (e.g., 'Your eyes are open.')
    """
    global alert_active, last_alert_time

    now = time.time()

    # Warm-up: avoid shouting in first few seconds
    if now - start_time < WARMUP_DURATION:
        return

    # Cooldown: avoid spamming
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
                # Custom simple message
                assistant.speak(custom_message)
            elif level is not None:
                # Use drowsiness-aware message builder in assistant
                assistant.alert_drowsiness(level, context)
            else:
                # Fallback
                assistant.speak("Please stay focused on the road.")
        finally:
            alert_active = False

    threading.Thread(target=_run, daemon=True).start()


# -------------------------------------------------
# 3. Main app: CNN + Haar + AI voice assistant integration
# -------------------------------------------------
def main():
    global invert_logic, closed_frames, start_time

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

    # Reset timers
    start_time = time.time()

    # 2. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[System] Error: Could not open webcam.")
        return
    assistant.speak(
        "Drowsiness detection assistant started. "
        "I will warn you if I detect signs of sleepiness."
    )

    # 3. Load Haar cascades (face + eyes)
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

        # Mirror for driver view
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(100, 100)
        )

        # Default overlay data
        cnn_status = "Unknown"
        is_cnn_drowsy = False

        for (x, y, w, h) in faces:
            # Draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # -------------------------
            # Mechanism 1: Eye detection
            # -------------------------
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, 1.1, 4, minSize=(20, 20)
            )

            if len(eyes) == 0:
                closed_frames += 1
            else:
                closed_frames = 0

            # Draw eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (0, 255, 0),
                    1,
                )

            # -------------------------
            # Mechanism 2: CNN inference
            # -------------------------
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

            # -------------------------
            # Consolidated alert logic
            # -------------------------
            final_status = "Scanning..."
            alert_color = (0, 255, 0)

            # Case A: Eyes Closed (highest priority)
            if closed_frames > EYE_CLOSED_FRAMES_THRESH:
                final_status = "EYES CLOSED!"
                alert_color = (0, 0, 255)
                # Very serious → HIGH alert
                trigger_voice_alert("high")

            # Case B: CNN says drowsy
            elif is_cnn_drowsy:
                final_status = f"CNN: {cnn_status}"
                alert_color = (0, 0, 255)
                # Medium/High drowsiness → MEDIUM level (you can change to "high")
                trigger_voice_alert("medium")

            # Case C: Active
            else:
                final_status = "Active"
                alert_color = (0, 255, 0)
                # Optional gentle info message (low priority)
                # If this is too chatty, comment the next line.
                trigger_voice_alert(
                    level=None,
                    custom_message="Your eyes are open. Please keep driving safely."
                )

            # Overlays
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

            # Warmup overlay
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

        # Show frame
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
