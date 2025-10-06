import os
import time
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load environment variables from .env
load_dotenv()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

def transcribe_full_audio(file_path: str) -> str:
    """Transcribe a full audio file (any length) using continuous recognition."""
    if not SPEECH_KEY or not SPEECH_REGION:
        raise ValueError("Azure Speech key or region is missing. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")

    try:
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        audio_config = speechsdk.audio.AudioConfig(filename=file_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    except Exception as e:
        print(f"Error setting up speech recognizer: {e}")
        return ""

    all_text = []
    done = False

    def recognized(evt):
        try:
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                all_text.append(evt.result.text)
        except Exception as e:
            print(f"Error processing recognized event: {e}")

    def stop(evt):
        nonlocal done
        done = True

    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop)
    recognizer.canceled.connect(stop)

    try:
        recognizer.start_continuous_recognition()
    except Exception as e:
        print(f"Error starting recognition: {e}")
        return ""

    try:
        while not done:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Recognition interrupted by user.")
    except Exception as e:
        print(f"Error during recognition loop: {e}")
    finally:
        try:
            recognizer.stop_continuous_recognition()
        except Exception as e:
            print(f"Error stopping recognition: {e}")

    return " ".join(all_text)


# if __name__ == "__main__":
#     file_path = r"C:\Users\vijay\Downloads\Record.wav"
#     try:
#         text = transcribe_full_audio(file_path)
#         if text:
#             print("Transcription result:\n", text)
#         else:
#             print("No text was recognized.")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
