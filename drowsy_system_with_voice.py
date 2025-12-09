# drowsy_system_with_voice.py

import time
import cv2
import joblib   # if you used scikit-learn to save the model

from voice_assistant import AIVoiceAssistant


# 1. Load ML model


MODEL_PATH = "models/drowsy_model.pkl"   # TODO:  real path eka danna

print("[System] Loading drowsiness model...")
model = joblib.load(MODEL_PATH)
print("[System] Model loaded.")


# 2. Create voice assistant


assistant = AIVoiceAssistant(
    driver_name="Praveen",
    use_cloud_assistant=True,           #  ONLINE ASSISTANT ENABLE
    gemini_model_name="gemini-2.5-flash"  # mekama ai studio list eken ena namakata match wenna one
)

# NOTE:
# - alert_drowsiness() still works fully offline (local TTS).
# - use_cloud_assistant=True mainly matters when you call
#   assistant.handle_command_with_nlp_backend(text) from
#   a mic script or some UI.


# Helper: feature extraction


def extract_features(frame):
    """
    TODO: Mehe oyage existing feature extraction logic danna:
      - eye aspect ratio (EAR)
      - mouth aspect ratio (MAR)
      - blink rate, yawn flag, etc.
    Should return a list/array matching what you used during training.
      e.g. [ear, mar, blink_rate, yawn_freq]
    """
    raise NotImplementedError("Implement extract_features() with your logic")


def get_speed():
    """TODO: Replace with real OBD/GPS speed if you have. For now constant."""
    return 60.0


def get_weather():
    """TODO: plug real weather source if available."""
    return "clear"



# 3. Main camera + detection loop


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

       
        # 3.1 Predict drowsiness
       
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

        
        # 3.2 Trigger voice assistant
        
        now = time.time()

        if level is not None and (now - last_alert_time) > ALERT_COOLDOWN:
            context = {
                "speed": get_speed(),
                "weather": get_weather(),
                "location_hint": "within a few kilometres",
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
