# voice_assistant.py

import os
import time
import pyttsx3
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AIVoiceAssistant:
    """
    AI Voice Assistant module for the drowsiness detection system.
    - Offline TTS using pyttsx3.
    - Context-aware drowsiness alerts (level + speed + weather + time).
    - Simple rule-based text commands (offline).
    - Optional OpenAI Assistants API integration for richer responses.
    """

    def __init__(
        self,
        driver_name: Optional[str] = None,
        language: str = "en",
        use_cloud_assistant: bool = False,
        assistant_id: Optional[str] = None,
    ) -> None:
        self.driver_name = driver_name or "driver"
        self.language = language

        # ---- Offline TTS setup ----
        self.engine = pyttsx3.init()
        # ---- Cloud assistant setup (optional) ----
        self.use_cloud_assistant = use_cloud_assistant and OPENAI_AVAILABLE
        self.assistant_id = assistant_id
        self.client: Optional["OpenAI"] = None
        self.assistant_id = assistant_id
        self.client: Optional[OpenAI] = None

        if self.use_cloud_assistant:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("[VoiceAssistant] OPENAI_API_KEY not found. Disabling cloud assistant.")
                self.use_cloud_assistant = False
            elif not self.assistant_id:
                print("[VoiceAssistant] assistant_id not provided. Disabling cloud assistant.")
                self.use_cloud_assistant = False
            else:
                try:
                    self.client = OpenAI()
                    print("[VoiceAssistant] Cloud assistant enabled.")
                except Exception as e:
                    print(f"[VoiceAssistant] Failed to init OpenAI client: {e}")
                    self.use_cloud_assistant = False

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _configure_engine(self) -> None:
        """Tune voice rate and volume (non-irritating but clear)."""
        try:
            rate = self.engine.getProperty("rate")
            self.engine.setProperty("rate", max(100, rate - 25))

            volume = self.engine.getProperty("volume")
            self.engine.setProperty("volume", min(1.0, volume + 0.2))

            # Optional: pick specific voice
            # voices = self.engine.getProperty("voices")
            # if voices:
            #     self.engine.setProperty("voice", voices[0].id)
        except Exception as e:
            print(f"[VoiceAssistant] Warning: could not configure engine: {e}")

    # -------------------------------------------------
    # Core TTS
    # -------------------------------------------------

    def speak(self, text: str) -> None:
        """Print + speak the given text (re-init engine each time to avoid silent bug)."""
        if not text:
            return

        print(f"[Assistant] {text}")

        try:
            # 🔁 create fresh engine for every utterance
            engine = pyttsx3.init("sapi5")  # or just pyttsx3.init() if you didn't force sapi5
            # basic config
            rate = engine.getProperty("rate")
            engine.setProperty("rate", max(100, rate - 25))
            engine.setProperty("volume", 1.0)

            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)

            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[VoiceAssistant] TTS error: {e}")


    # -------------------------------------------------
    # Drowsiness-related logic
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
            f"{name}, I detected signs of drowsiness. Please stay alert."
        )

        # --- Add contextual info ---
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

    # -------------------------------------------------
    # Offline rule-based command handling
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Cloud / Assistants API-backed handling (optional)
    # -------------------------------------------------

    def _ask_cloud_assistant(self, user_text: str) -> Optional[str]:
        """
        Send text to OpenAI Assistants API and return response text, or None on failure.
        """
        if not (self.use_cloud_assistant and self.client and self.assistant_id):
            return None

        try:
            # 1. Create a new thread
            thread = self.client.beta.threads.create()

            # 2. Add user's message
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_text,
            )

            # 3. Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id,
            )

            # 4. Poll until completed or failed
            while True:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id,
                )
                if run.status in ("completed", "failed", "cancelled", "expired"):
                    break
                time.sleep(0.5)

            if run.status != "completed":
                return None

            # 5. Get the latest assistant message
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            latest = messages.data[0]

            text_parts = []
            for part in latest.content:
                if part.type == "text":
                    text_parts.append(part.text.value)

            answer = " ".join(text_parts).strip()
            return answer or None
        
        except Exception as e:
            print(f"[VoiceAssistant] Error in cloud assistant: {e}")
            print("==== CLOUD ASSISTANT ERROR ====")
            print(repr(e))
            print("===============================")
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
