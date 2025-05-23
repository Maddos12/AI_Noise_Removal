import os
import numpy as np
import librosa
import pickle
from moviepy import VideoFileClip
import speech_recognition as sr
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_DIR = "audio_output"
DATASET_DIR = "dataset"
ENCODING_FILE = "accent_encodings.pkl"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_audio(video_path):
    """Extract audio from video and save as WAV."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError("File not found: " + video_path)

    audio_path = os.path.join(UPLOAD_DIR, "output.wav")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()
    return audio_path

def transcribe_audio(audio_path):
    """Transcribe audio to text using Google's speech API."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "[Unintelligible Audio]"
    except sr.RequestError:
        return "[Error contacting speech service]"

def detect_language(text):
    """Detect language using langdetect."""
    try:
        return detect(text)
    except Exception:
        return "Could not detect language"

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f" Error processing {file_path}: {e}")
        return None

def build_accent_encodings(dataset_dir):
    """Create average feature vector for each accent category."""
    encodings = {}

    for accent in os.listdir(dataset_dir):
        accent_path = os.path.join(dataset_dir, accent)
        if not os.path.isdir(accent_path):
            continue

        print(f" Processing accent: {accent}")
        encodings[accent] = []

        for file in os.listdir(accent_path):
            if file.endswith(".wav") or file.endswith(".mp3"):
                file_path = os.path.join(accent_path, file)
                features = extract_features(file_path)
                if features is not None:
                    encodings[accent].append(features)

    accent_profiles = {
        accent: np.mean(features, axis=0)
        for accent, features in encodings.items() if features
    }

    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(accent_profiles, f)

    print(f"\n Accent encoding file saved as: {ENCODING_FILE}")

def load_accent_encodings(file_path=ENCODING_FILE):
    """Load prebuilt accent encodings."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Accent encodings file not found. Run 'build' first.")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def compare_to_known_accents(audio_path, accent_encodings):
    """Extract MFCC from audio and compare with stored accent averages."""
    user_features = extract_features(audio_path)
    if user_features is None:
        return {}

    user_features = user_features.reshape(1, -1)

    similarities = {}
    for accent, known_vector in accent_encodings.items():
        known_vector = known_vector.reshape(1, -1)
        similarity = cosine_similarity(user_features, known_vector)[0][0]
        similarity_percent = round(similarity * 100, 2)
        similarities[accent] = similarity_percent

    return similarities

def main_loop():
    print(" AI Audio Analyzer")
    print("Type a video file path to analyze, 'build' to rebuild accent dataset, or 'exit' to quit.\n")

    while True:
        user_input = input(" Enter video path, 'build', or 'exit': ").strip()

        if user_input.lower() == "exit":
            print(" Exiting program.")
            break

        elif user_input.lower() == "build":
            build_accent_encodings(DATASET_DIR)
            break

        elif not os.path.isfile(user_input):
            print(" File not found. Try again.")
            continue

        try:
            print(" Extracting audio...")
            audio_path = extract_audio(user_input)
            print(f" Audio saved at: {audio_path}")

            print(" Transcribing speech...")
            text = transcribe_audio(audio_path)
            print(f" Transcribed Text:\n{text}")

            print(" Detecting language...")
            lang = detect_language(text)
            print(f" Detected Language: {lang}")

            if lang == "en":
                if any(w in text.lower() for w in ["mate", "bloody", "innit"]):
                    print(" Likely British English")
                elif any(w in text.lower() for w in ["y'all", "gonna", "gotta"]):
                    print(" Likely American English")
                else:
                    print(" Possibly other English (e.g., AUS, IN, CA)")
            else:
                print(" Accent detection not available for non-English.")

            # Compare to known accents if encodings file exists
            if os.path.exists(ENCODING_FILE):
                print(" Comparing accent with known profiles...")
                accent_encodings = load_accent_encodings()
                similarities = compare_to_known_accents(audio_path, accent_encodings)

                for accent, score in similarities.items():
                    print(f"   - {accent.title()}: {score}% match")

                best_match = max(similarities, key=similarities.get)
                if similarities[best_match] >= 90:
                    print(f" Accent confidently identified as: {best_match.title()}")
                else:
                    print(" Accent not confidently recognized.")
            else:
                print("⚠ No accent dataset found. Run 'build' to generate it.")

        except Exception as e:
            print(f" Error: {e}")

        print("\n---\n")

if __name__ == '__main__':
    main_loop()
