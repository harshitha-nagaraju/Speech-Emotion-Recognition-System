import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pyttsx3
import speech_recognition as sr

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error extracting features:", file_path, e)
        return None
        
def load_dataset(dataset_path):
    features = []
    emotions = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    emotion_label = int(file.split("-")[2])
                except:
                    continue
                
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    emotions.append(emotion_label)
    
    return np.array(features), np.array(emotions)

def record_audio(filename="my_voice.wav", duration=4, fs=44100):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)
    print(f"Recording saved as '{filename}'")

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        try:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print(f"Transcription: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError:
            print("Speech recognition service error.")
    return ""

def predict_emotion(model, encoder, filename):
    features = extract_features(filename)
    if features is None:
        print("Could not extract features.")
        return
    features = features.reshape(1, -1)
    prediction = model.predict(features)

    emotion_map = {
        1: "Neutral",
        2: "Calm",
        3: "Happy",
        4: "Sad",
        5: "Angry",
        6: "Fearful",
        7: "Disgust",
        8: "Surprised"
    }

    emotion = emotion_map.get(prediction[0], "Unknown")

    spoken_text = transcribe_audio(filename)
    message = f"You said: {spoken_text}. The predicted emotion is: {emotion}"
    print(message)

    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()


if __name__ == "__main__":
    dataset_path = r"C:\Users\harsh\Downloads\ravdess" # Replace with your actual dataset path

    print("Loading dataset...")
    X, y = load_dataset(dataset_path)

    if len(X) == 0:
        print("Dataset not loaded properly. Check your path.")
        exit()

    print("Encoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print(f"Model accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")

    choice = input("Choose:\n1. Record your voice\n2. Enter audio file path\nEnter 1 or 2: ").strip()

    if choice == "1":
        record_audio("my_voice.wav")
        predict_emotion(model, encoder, "my_voice.wav")
    elif choice == "2":
        audio_path = input("Enter path to audio file (.wav): ").strip('"').strip("'")
        if os.path.exists(audio_path):
            predict_emotion(model, encoder, audio_path)
        else:
            print("File does not exist.")
    else:
        print("Invalid input.")
