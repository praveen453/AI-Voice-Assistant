# drowsy_system_with_voice.py

import time
import cv2
import joblib   # or from tensorflow import keras etc.

from voice_assistant import AIVoiceAssistant

# =========================
# 1. Load ML model
# =========================

MODEL_PATH = "models/drowsy_model.pkl"   # <<< oyāge real path eka dapan

print("[System] Loading drowsiness model...")
model = joblib.load(MODEL_PATH)
print("[System] Model loaded.")

# =========================
# 2. Create voice assistant
# =========================

assistant = AIVoiceAssistant(driver_name="Praveen")

# =========================
# Helper: feature extraction
# =========================

def extract_features(frame):
    """
    TODO: Mehe oyāge existing feature extraction logic danna:
      - eye aspect ratio (EAR)
      - mouth aspect ratio (MAR)
      - blink rate, yawn flag etc.
    For demo, dummy list ekak return karanawa.
    """
    # EXAMPLE ONLY:
    # return [ear_value, mar_value, blink_count, yawn_count]
    raise NotImplementedError("Implement extract_features() with your logic")


def get_speed():
    """
    TODO: If you have OBD-II / GPS speed, return km/h.
    Neththam simply 60 wage constant value ekak denna puluwan.
    """
    return 60.0


def get_weather():
    """
    TODO: If you have weather source, plug it.
    Neththam 'clear' kiyala constant ekak use karapan.
    """
    return "clear"


# =========================
# 3. Main camera + detection loop
# =========================

cap = cv2.VideoCapture(0)

last_alert_time = 0.0
ALERT_COOLDOWN = 30.0   # seconds between alerts

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[System] Could not read frame from camera.")
            break

        try:
            features = extract_features(frame)
        except NotImplementedError:
            print("[System] extract_features() not implemented yet.")
            break

        # -------------------------------
        # 3.1 Predict drowsiness
        # -------------------------------
        # Example for sklearn model with predict_proba
        prob_sleepy = model.predict_proba([features])[0][1]  # sleepy class prob

        # Map probability -> level
        if prob_sleepy >= 0.9:
            level = "high"
        elif prob_sleepy >= 0.7:
            level = "medium"
        elif prob_sleepy >= 0.5:
            level = "low"
        else:
            level = None  # not drowsy

        # -------------------------------
        # 3.2 Trigger voice assistant
        # -------------------------------
        now = time.time()

        if level is not None and (now - last_alert_time) > ALERT_COOLDOWN:
            context = {
                "speed": get_speed(),
                "weather": get_weather(),
                "location_hint": "within a few kilometres",  # optional
            }

            print(f"[System] Drowsiness detected: level={level}, prob={prob_sleepy:.2f}")
            assistant.alert_drowsiness(level, context)

            last_alert_time = now

        # Optional: show frame for debugging
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
