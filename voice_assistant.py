# voice_assistant.py

import os
import time
import pyttsx3
from datetime import datetime
from typing import Optional, Dict, Any

# Gemini cloud assistant (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # type: ignore


class AIVoiceAssistant:
    """
    AI Voice Assistant module for the drowsiness detection system.

    - Offline TTS using pyttsx3.
    - Context-aware drowsiness alerts (level + speed + weather + time).
    - Simple rule-based text commands (offline).
    - Optional Gemini cloud integration for richer responses.
    """

    def __init__(
        self,
        driver_name: Optional[str] = None,
        language: str = "en",
        use_cloud_assistant: bool = False,
        gemini_model_name: str = "gemini-2.5-flash",
    ) -> None:
        self.driver_name = driver_name or "driver"
        self.language = language

        # Offline TTS setup (ONE shared engine) 
        self.engine = pyttsx3.init("sapi5")
        self._configure_engine()

        # Cloud assistant setup (Gemini, optional)
        self.use_cloud_assistant = use_cloud_assistant and GEMINI_AVAILABLE
        self.gemini_model_name = gemini_model_name
        self.gemini_model = None

        if self.use_cloud_assistant:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("[VoiceAssistant] GEMINI_API_KEY not found. Disabling cloud assistant.")
                self.use_cloud_assistant = False
            else:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                    print(
                        f"[VoiceAssistant] Gemini cloud assistant enabled "
                        f"({self.gemini_model_name})."
                    )
                except Exception as e:
                    print(f"[VoiceAssistant] Failed to init Gemini model: {e}")
                    self.use_cloud_assistant = False

    
    # Internal helpers
    

    def _configure_engine(self) -> None:
        """Tune voice rate and volume (non-irritating but clear)."""
        try:
            rate = self.engine.getProperty("rate")
            self.engine.setProperty("rate", max(100, rate - 25))

            self.engine.setProperty("volume", 1.0)

            voices = self.engine.getProperty("voices")
            if voices:
                self.engine.setProperty("voice", voices[0].id)
        except Exception as e:
            print(f"[VoiceAssistant] Warning: could not configure engine: {e}")

  
    # Core TTS
    

    def speak(self, text: str) -> None:
        """
        Print + speak the given text.
        Use one shared pyttsx3 engine and restart it if something goes wrong.
        """
        if not text:
            return

        print(f"[Assistant] {text}")

        try:
            # stop any previous utterance (just in case)
            self.engine.stop()
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[VoiceAssistant] TTS error on first try: {e}")
            # try re-create engine once
            try:
                self.engine = pyttsx3.init("sapi5")
                self._configure_engine()
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e2:
                print(f"[VoiceAssistant] TTS error after re-init: {e2}")

    
    # Drowsiness-related logic
    

    def build_drowsiness_message(
        self,
        level: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a context-aware message based on drowsiness level and context.
        level: "low", "medium", "high"
        context keys (optional):
          - speed (km/h)
          - weather ("rainy", "foggy", "clear", etc.)
          - location_hint (string like "near town", optional)
        """
        context = context or {}
        name = self.driver_name

        base_messages = {
            "low": (
                f"{name}, I noticed early signs of tiredness. "
                "Please stay focused on the road."
            ),
            "medium": (
                f"{name}, you seem drowsy. "
                "Please slow down and consider taking a short break soon."
            ),
            "high": (
                f"{name}, you are extremely drowsy. "
                "Please pull over at a safe place immediately and rest."
            ),
        }

        message = base_messages.get(
            level.lower(),
            f"{name}, I detected signs of drowsiness. Please stay alert."
        )

        # Add contextual info 
        speed = context.get("speed")
        weather = context.get("weather")
        location_hint = context.get("location_hint")

        if speed is not None:
            try:
                speed_val = float(speed)
                if speed_val >= 80:
                    message += " You are travelling at high speed, be extra careful."
                elif speed_val >= 40:
                    message += " You are currently moving; please drive carefully."
                else:
                    message += " Your speed is low, but please find a safe place to rest."
            except ValueError:
                pass

        if weather:
            if str(weather).lower() in {"rainy", "foggy", "stormy"}:
                message += " Road conditions are not ideal due to the weather."

        if location_hint:
            message += f" There may be rest stops {location_hint}."

        # Time-based reminder
        hour = datetime.now().hour
        if hour >= 23 or hour <= 5:
            message += " It is late at night, fatigue risk is higher."

        return message

    def alert_drowsiness(
        self,
        level: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Public method for the drowsiness detection module.
        Example:
            assistant.alert_drowsiness("medium", {"speed": 75, "weather": "rainy"})
        """
        msg = self.build_drowsiness_message(level, context)
        self.speak(msg)

    # Offline rule-based command handling
   

    def handle_text_command(self, user_text: str) -> None:
        """
        Simple rule-based commands (offline).
        """
        original_text = user_text
        user_text = (user_text or "").lower().strip()

        if not user_text:
            self.speak("I did not hear any command.")
            return

        # Exit / stop assistant
        if "exit" in user_text or "quit" in user_text or "stop assistant" in user_text:
            self.speak("Stopping the assistant now. Drive safely.")
            raise SystemExit

        if "how am i" in user_text or "status" in user_text:
            self.speak(
                "Your current status will be based on drowsiness detection and vehicle data. "
                "Right now, this is a prototype running on your computer."
            )
        elif "test drowsy" in user_text or "simulate drowsiness" in user_text:
            self.speak("Okay, I will simulate a drowsiness alert now.")
            self.alert_drowsiness("medium", {"speed": 60, "weather": "clear"})
        elif "play music" in user_text:
            self.speak(
                "This prototype cannot play music yet, but the final system will support media control."
            )
        elif "nearest" in user_text and "rest" in user_text:
            self.speak(
                "In the final system, I will show nearby rest stops on the map and give you directions."
            )
        elif "hello" in user_text or "hi" in user_text:
            self.speak("Hello! I am your smart drowsiness assistant. How can I help you?")
        elif "time" in user_text:
            now = datetime.now().strftime("%H:%M")
            self.speak(f"The current time is {now}.")
        else:
            self.speak(
                f"Sorry, I do not understand the command '{original_text}' yet. "
                "This is only a prototype."
            )

   
    # Cloud / Gemini-backed handling (optional)
   

    def _ask_cloud_assistant(self, user_text: str) -> Optional[str]:
        """
        Send text to Gemini and return response text, or None on failure.
        """
        if not (self.use_cloud_assistant and self.gemini_model):
            return None

        try:
            prompt = (
                "You are a driving safety and drowsiness assistant. "
                "Give short, clear spoken responses suitable to read aloud to a driver. "
                "Avoid long paragraphs. "
                "Now answer this:\n\n"
                f"Driver said: {user_text}"
            )

            response = self.gemini_model.generate_content(prompt)

            # Prefer response.text if available
            text = getattr(response, "text", None)
            if text:
                return text.strip()

            # Fallback: use first candidate parts
            if hasattr(response, "candidates") and response.candidates:
                parts = response.candidates[0].content.parts
                joined = " ".join(getattr(p, "text", "") for p in parts)
                joined = joined.strip()
                if joined:
                    return joined

            return None

        except Exception as e:
            print("==== GEMINI CLOUD ERROR ====")
            print(repr(e))
            print("================================")
            return None

    def handle_command_with_nlp_backend(self, user_text: str) -> None:
        """
        Entry point for mic / text input.
        If cloud assistant is enabled, try it first.
        Otherwise, or on failure, fall back to offline rules.
        """
        if not self.use_cloud_assistant:
            self.handle_text_command(user_text)
            return

        response = self._ask_cloud_assistant(user_text)
        if response:
            self.speak(response)
        else:
            self.speak(
                "I could not reach the cloud assistant. I will use my offline commands instead."
            )
            self.handle_text_command(user_text)
