from pydub import AudioSegment
import speech_recognition as sr
import nltk
import os
from deepmultilingualpunctuation import PunctuationModel

# Ensure you have the Punkt tokenizer data downloaded
nltk.download('punkt')

# Initialize the punctuation restoration model
punctuation_model = PunctuationModel()

def convert_to_wav(input_path, wav_path):
    audio = AudioSegment.from_file(input_path)  # Detects format based on file extension
    audio.export(wav_path, format="wav")

def audio_file_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            print("Full Transcription without punctuation:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None
        except sr.RequestError:
            print("Could not request results from the speech recognition service; check your network connection")
            return None

def restore_punctuation(text):
    return punctuation_model.restore_punctuation(text)

def print_sentences(transcription):
    if transcription:
        # Restore punctuation to transcription
        punctuated_text = restore_punctuation(transcription)
        print("\nPunctuated Transcription:", punctuated_text)

        # Split text into sentences
        sentences = nltk.sent_tokenize(punctuated_text)
        print("\nTranscription in sentences:")
        for i, sentence in enumerate(sentences, 1):
            print(f"Sentence {i}: \"{sentence}\"")

# Paths for input and temporary WAV output
input_path = r'D:\Be mini-project\audios\maneaudio.mp3'  # Replace with your input file path
wav_path = r'D:\Be mini-project\audios\maneaudio.wav'


# Convert the audio file to WAV format
convert_to_wav(input_path, wav_path)

# Transcribe the WAV file
transcription = audio_file_to_text(wav_path)

# Print each sentence separately after restoring punctuation
print_sentences(transcription)

# Optionally, delete the temporary WAV file after processing
os.remove(wav_path)