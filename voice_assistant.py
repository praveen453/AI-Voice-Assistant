# voice_assistant.py

import os
import time
import pyttsx3

from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()

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

    - Offline TTS using Windows System.Speech.
    - Context-aware drowsiness alerts.
    - Simple offline commands.
    - Gemini cloud Q&A support (auto fallback).
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

        # Keep pyttsx3 engine (backup)
        self.engine = pyttsx3.init("sapi5")
        self._configure_engine()

        # Cloud (Gemini)
        self.use_cloud_assistant = use_cloud_assistant and GEMINI_AVAILABLE
        self.gemini_model_name = gemini_model_name
        self.gemini_model = None

        if self.use_cloud_assistant:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("[VoiceAssistant] GEMINI_API_KEY not found. Cloud disabled.")
                self.use_cloud_assistant = False
            else:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                    print(f"[VoiceAssistant] Gemini cloud assistant enabled ({self.gemini_model_name}).")
                except Exception as e:
                    print(f"[VoiceAssistant] Failed to init Gemini model: {e}")
                    self.use_cloud_assistant = False

 
    # Offline voice settings
   
    def _configure_engine(self) -> None:
        try:
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 1.0)
        except Exception as e:
            print(f"[VoiceAssistant] Could not configure TTS: {e}")

    
    # SPEAK FUNCTION (Windows System.Speech)
   
    def speak(self, text: str) -> None:
        if not text:
            return

        print(f"[Assistant] {text}")

        try:
            import subprocess, json
            ps_text = json.dumps(text)

            cmd = [
                "powershell",
                "-Command",
                (
                    "Add-Type -AssemblyName System.Speech;"
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                    "$s.Rate = 0; $s.Volume = 100;"
                    f"$s.Speak({ps_text});"
                ),
            ]

            subprocess.run(cmd, check=False)

        except Exception as e:
            print(f"[VoiceAssistant] TTS error: {e}")


    # BUILD DROWSINESS MESSAGE
 
    def build_drowsiness_message(self, level: str, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}
        name = self.driver_name

        base = {
            "low":   f"{name}, I noticed early signs of tiredness.",
            "medium": f"{name}, you seem drowsy. Please slow down and consider taking a short break soon.",
            "high":   f"{name}, you are extremely drowsy. Please pull over at a safe place immediately and rest.",
        }

        msg = base.get(level.lower(), f"{name}, please stay alert.")

        if "speed" in context:
            try:
                spd = float(context["speed"])
                if spd >= 80:
                    msg += " You are moving fast, be extra careful."
                elif spd >= 40:
                    msg += " You are currently moving; please drive carefully."
            except:
                pass

        if context.get("weather") in ["rainy", "foggy", "stormy"]:
            msg += " Road conditions are not ideal."

        if context.get("location_hint"):
            msg += f" There may be rest stops {context['location_hint']}."

        hour = datetime.now().hour
        if hour >= 23 or hour <= 5:
            msg += " It is late at night; fatigue risk is higher."

        return msg

  
    # ALERT DROWSINESS
    
    def alert_drowsiness(self, level: str, context: Optional[Dict[str, Any]] = None) -> None:
        msg = self.build_drowsiness_message(level, context)
        self.speak(msg)

    
    # OFFLINE COMMAND HANDLER
   
    def handle_text_command(self, user_text: str) -> None:
        t = (user_text or "").lower().strip()

        if "exit" in t or "quit" in t:
            self.speak("Stopping the assistant. Drive safely.")
            raise SystemExit

        if "hello" in t:
            self.speak("Hello! How can I help you?")
        elif "time" in t:
            now = datetime.now().strftime("%H:%M")
            self.speak(f"The time is {now}.")
        elif "test" in t:
            self.alert_drowsiness("medium", {"speed": 60})
        else:
            self.speak("I did not understand that command.")

    
    # CLOUD NLP (Gemini)
    
    def _ask_cloud_assistant(self, user_text: str) -> Optional[str]:
        if not (self.use_cloud_assistant and self.gemini_model):
            return None

        try:
            prompt = (
                "You are an AI driving assistant helping a sleepy driver. "
                "Give short, clear spoken answers.\n"
                f"Driver said: {user_text}"
            )

            response = self.gemini_model.generate_content(prompt)

            if hasattr(response, "text"):
                return response.text.strip()

            if hasattr(response, "candidates"):
                parts = response.candidates[0].content.parts
                out = " ".join(getattr(p, "text", "") for p in parts).strip()
                return out or None

            return None

        except Exception as e:
            print("==== GEMINI ERROR ====")
            print(e)
            print("======================")
            return None

   
    # MAIN NLP ENTRY (ONLINE → OFFLINE FALLBACK)

    def handle_command_with_nlp_backend(self, user_text: str) -> None:
        t = (user_text or "").lower().strip()

        if "exit" in t:
            self.handle_text_command(t)
            return

        if not self.use_cloud_assistant:
            self.handle_text_command(t)
            return

        # Try online
        response = self._ask_cloud_assistant(user_text)

        if response:
            self.speak(response)
        else:
            self.speak("Online assistant unavailable. Switching to offline mode.")
            self.handle_text_command(user_text)
