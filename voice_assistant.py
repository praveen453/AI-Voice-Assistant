# voice_assistant.py

import os
import time
from datetime import datetime
from typing import Optional, Dict, Any
import subprocess


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

    - Offline TTS using Windows System.Speech (via PowerShell).
    - Context-aware drowsiness alerts (level + speed + weather + time).
    - Simple rule-based text commands (offline).
    - Optional Gemini cloud integration for richer responses.
      *Both drowsiness alerts and free chat can use the API.*
      If API fails, system falls back to offline messages.
    """

    def __init__(
        self,
        driver_name: Optional[str] = None,
        language: str = "en",
        use_cloud_assistant: bool = False,
        gemini_model_name: str = "models/gemini-2.5-flash",
    ) -> None:
        self.driver_name = driver_name or "driver"
        self.language = language

        # ---- Cloud assistant setup (Gemini, optional) ----
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

    # -------------------------------------------------
    # Core TTS
    # -------------------------------------------------



def speak(self, text: str) -> None:
    """Use Microsoft Edge TTS for stable voice output."""
    if not text:
        return

    print(f"[Assistant] {text}")

    try:
        output_file = f"tts_{uuid.uuid4().hex}.mp3"

        subprocess.run([
            "python", "-m", "edge_tts",
            "--voice", "en-US-GuyNeural",
            "--text", text,
            "--write-media", output_file
        ], check=True)

        # Play audio
        if os.path.exists(output_file):
            os.system(f'start {output_file}')

    except Exception as e:
        print(f"[VoiceAssistant] EDGE-TTS error: {e}")


    # -------------------------------------------------
    # OFFLINE drowsiness message builder
    # -------------------------------------------------

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
            f"{name}, I detected signs of drowsiness. Please stay alert.",
        )

        # Contextual information
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

        hour = datetime.now().hour
        if hour >= 23 or hour <= 5:
            message += " It is late at night, fatigue risk is higher."

        return message

    # -------------------------------------------------
    # CLOUD (Gemini) drowsiness message builder
    # -------------------------------------------------

    def _build_drowsiness_message_cloud(
        self,
        level: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        If cloud assistant is enabled, ask Gemini to generate a spoken
        warning based on drowsiness level + context.
        Returns text or None on failure.
        """
        if not (self.use_cloud_assistant and self.gemini_model):
            return None

        context = context or {}
        name = self.driver_name

        level_text = {
            "low": "mild drowsiness",
            "medium": "moderate drowsiness",
            "high": "severe and dangerous drowsiness",
        }.get(level.lower(), "some drowsiness")

        speed = context.get("speed")
        weather = context.get("weather")
        location_hint = context.get("location_hint")

        parts = [f"The driver is {name}.",
                 f"The system detected {level_text} while driving."]

        if speed is not None:
            parts.append(f"The estimated speed is {speed} kilometers per hour.")
        if weather:
            parts.append(f"The weather is {weather}.")
        if location_hint:
            parts.append(f"There may be rest areas {location_hint}.")

        parts.append(
            "Generate a short, clear warning sentence to speak to the driver. "
            "Focus on safety, do not give long explanations."
        )

        prompt = " ".join(parts)

        try:
            response = self.gemini_model.generate_content(prompt)

            text = getattr(response, "text", None)
            if text:
                return text.strip()

            if hasattr(response, "candidates") and response.candidates:
                segs = []
                for p in response.candidates[0].content.parts:
                    segs.append(getattr(p, "text", ""))
                out = " ".join(segs).strip()
                if out:
                    return out

            return None

        except Exception as e:
            print("==== GEMINI DROWSY ERROR ====")
            print(repr(e))
            print("================================")
            return None

    # -------------------------------------------------
    # Public drowsiness alert API (used by ML model)
    # -------------------------------------------------

    def alert_drowsiness(
        self,
        level: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Called by the drowsiness detection module.

        Behaviour:
        - If cloud assistant is on → try Gemini to generate alert text.
        - If API fails or is disabled → use offline template.
        - Then speak the final message.
        """
        msg: Optional[str] = None

        # 1) Try cloud (Gemini)
        if self.use_cloud_assistant:
            msg = self._build_drowsiness_message_cloud(level, context)

        # 2) Fallback to offline template if cloud not used / failed
        if not msg:
            msg = self.build_drowsiness_message(level, context or {})

        # 3) Speak the final text
        self.speak(msg)

    # -------------------------------------------------
    # Offline rule-based commands (for mic / text chat)
    # -------------------------------------------------

    def handle_text_command(self, user_text: str) -> None:
        """
        Simple rule-based commands (offline).
        Used when:
          - Cloud assistant is disabled, OR
          - Cloud assistant failed, OR
          - We intentionally want offline behaviour.
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

    # -------------------------------------------------
    # Generic cloud Q&A (for mic text)
    # -------------------------------------------------

    def _ask_cloud_assistant(self, user_text: str) -> Optional[str]:
        """
        Generic Q&A via Gemini (driver questions).
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

            text = getattr(response, "text", None)
            if text:
                return text.strip()

            if hasattr(response, "candidates") and response.candidates:
                parts = response.candidates[0].content.parts
                joined = " ".join(getattr(p, "text", "") for p in parts)
                joined = joined.strip()
                if joined:
                    return joined

            return None

        except Exception as e:
            print("==== GEMINI CHAT ERROR ====")
            print(repr(e))
            print("================================")
            return None

    def handle_command_with_nlp_backend(self, user_text: str) -> None:
        """
        Entry point for mic / text input.

        1) First handle local control commands (exit / quit etc.).
        2) If cloud assistant is enabled, try Gemini Q&A.
        3) On failure or when cloud is disabled, fall back to offline rules.
        """
        text_norm = (user_text or "").lower().strip()

        # 1. Local control commands (always offline, so exit works even offline)
        if "exit" in text_norm or "quit" in text_norm or "stop assistant" in text_norm:
            self.handle_text_command(user_text)
            return

        # 2. If no cloud assistant, use offline rules directly
        if not self.use_cloud_assistant:
            self.handle_text_command(user_text)
            return

        # 3. Try Gemini (online)
        response = self._ask_cloud_assistant(user_text)

        if response:
            self.speak(response)
        else:
            self.speak(
                "I could not reach the online assistant. "
                "I will use my offline commands instead."
            )
            self.handle_text_command(user_text)
