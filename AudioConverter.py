from pydub import AudioSegment
import os

# Allowed audio extensions
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}

def convert_to_wav(input_file: str, output_file: str = None) -> str:

    try:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        ext = os.path.splitext(input_file)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File format '{ext}' is not supported.")

        if ext == ".wav" and output_file is None:
            print("Input file is already WAV. Skipping conversion.")
            return input_file

        if output_file is None:
            base, _ = os.path.splitext(input_file)
            output_file = base + ".wav"

        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        print(f"File converted successfully: {output_file}")
        return output_file

    except Exception as e:
        print(f"Error converting file: {e}")
        return None


# Example usage
# convert_to_wav(r"C:\Users\vijay\Downloads\Record.mp3")
