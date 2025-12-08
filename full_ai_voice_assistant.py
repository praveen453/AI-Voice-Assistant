# full_ai_voice_assistant_mic.py

import time
import speech_recognition as sr
from voice_assistant import AIVoiceAssistant

# 🔴 VERY IMPORTANT:
# Set this to the SAME mic index that worked in your mic_test.py
MIC_INDEX = 3  # <-- CHANGE THIS


def listen_for_command(
    recognizer: sr.Recognizer,
    microphone: sr.Microphone,
) -> str:
    """
    Listen from the selected microphone and return recognized text.
    Uses Google's speech API (needs internet for STT).
    """
    with microphone as source:
        print("\n[System] Listening... Please speak your command.")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        audio = recognizer.listen(source)

    print("[System] Got audio, recognizing...")

    try:
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"[You] {text}")
        return text
    except sr.UnknownValueError:
        print("[System] Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"[System] Speech recognition service error: {e}")
        return ""


def main():
    print("=== FULL AI VOICE ASSISTANT (MIC + API VERSION) ===")
    print("Press Ctrl + C to stop.\n")

    # ---- Configure assistant here ----
    # 1) If you ONLY want offline behaviour (no OpenAI):
    # assistant = AIVoiceAssistant(driver_name="Nimal")

    # 2) If you want to use OpenAI Assistants API:
    assistant = AIVoiceAssistant(
        driver_name="praveen",
        use_cloud_assistant=True,              # must be True to use API
        assistant_id="asst_52I4wLfLNGoST6knapDwAwO7"  # TODO: replace with your real ID
    )

    assistant.speak(
        "Hello. I am your smart drowsiness assistant using microphone input. "
        "When I say listening, speak in English. "
        "For example, say 'hello', 'how am I', 'test drowsy', or 'exit assistant'."
    )

    recognizer = sr.Recognizer()

    # Show microphones and create the selected device
    microphones = sr.Microphone.list_microphone_names()
    print("Available microphones:")
    for i, name in enumerate(microphones):
        print(f"  {i}: {name}")

    print(f"\n[System] Using MIC_INDEX = {MIC_INDEX}")
    try:
        microphone = sr.Microphone(device_index=MIC_INDEX)
    except OSError as e:
        print(f"[System] Could not open microphone index {MIC_INDEX}: {e}")
        assistant.speak(
            "I could not access the microphone. Please check your audio settings and try again."
        )
        return

    while True:
        try:
            user_text = listen_for_command(recognizer, microphone)

            if not user_text:
                assistant.speak("I did not catch that. Please repeat your command.")
                continue

            # This calls cloud assistant if enabled, else offline commands
            assistant.handle_command_with_nlp_backend(user_text)

            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[System] Keyboard interrupt detected. Exiting.")
            assistant.speak("Stopping the assistant now. Goodbye.")
            break
        except SystemExit:
            print("[System] Exit command received. Exiting loop.")
            break
        except Exception as e:
            print(f"[System] Unexpected error: {e}")
            assistant.speak("An error occurred. I will try to continue.")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
