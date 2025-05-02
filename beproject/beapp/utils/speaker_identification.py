import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import KMeans
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import wave
from datetime import timedelta
import moviepy as mp
import openai
class SpeakerIdentification:
    def __init__(self, video_path, min_silence_len=500, silence_thresh=-40):
        self.video_path = video_path
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.recognizer = sr.Recognizer()
        self.temp_audio_path = "temp_audio.wav"
        self.api_key = "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"

    def extract_audio_from_video(self):
        """Extract audio from MP4 video"""
        try:
            print("Extracting audio from video...")
            video = mp.VideoFileClip(self.video_path)
            video.audio.write_audiofile(self.temp_audio_path)
            return self.temp_audio_path
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            raise

    def split_audio_segments(self):
        """Split audio into segments based on silence"""
        self.extract_audio_from_video()
        print("Splitting audio into segments...")
        audio = AudioSegment.from_wav(self.temp_audio_path)
        chunks = split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=100
        )
        self.chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_path = f"temp_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            self.chunk_files.append(chunk_path)
        return self.chunk_files

    def extract_features(self, audio_path):
        """Extract MFCC features from audio"""
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)

    def identify_speakers(self, n_speakers=2):
        """Identify different speakers using clustering"""
        print("Identifying speakers...")
        features = [self.extract_features(chunk) for chunk in self.chunk_files]
        kmeans = KMeans(n_clusters=n_speakers, random_state=42)
        return kmeans.fit_predict(features)

    def get_timestamp(self, sample_duration, current_index):
        """Convert duration to timestamp format"""
        total_seconds = int(sample_duration * current_index)
        return str(timedelta(seconds=total_seconds))

    def transcribe_segments(self, speaker_labels):
        """Transcribe each audio segment and assign to speakers"""
        print("Transcribing segments...")
        transcripts = []
        for i, (chunk_path, speaker) in enumerate(zip(self.chunk_files, speaker_labels)):
            try:
                with sr.AudioFile(chunk_path) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                    with wave.open(chunk_path, 'rb') as wav_file:
                        duration = wav_file.getnframes() / wav_file.getframerate()
                        timestamp = self.get_timestamp(duration, i)
                    transcripts.append({
                        'speaker': f'Speaker {speaker + 1}',
                        'text': text,
                        'timestamp': timestamp,
                        'duration': str(timedelta(seconds=int(duration)))
                    })
            except sr.UnknownValueError:
                print(f"Could not understand audio in segment {i}")
            except sr.RequestError as e:
                print(f"Could not request results from speech recognition service; {e}")
        return transcripts

    def cleanup(self):
        """Remove temporary files"""
        print("Cleaning up temporary files...")
        if os.path.exists(self.temp_audio_path):
            os.remove(self.temp_audio_path)
        for chunk_path in self.chunk_files:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    def process_video(self, n_speakers=2):
        """Process the complete video file"""
        try:
            self.split_audio_segments()
            speaker_labels = self.identify_speakers(n_speakers)
            return self.transcribe_segments(speaker_labels)
        finally:
            self.cleanup()
        
    def fact_check_statement(self, statement):
        """Use OpenAI GPT-4 to fact-check the statement."""
        prompt = f"""
        Fact-check the following statement using credible sources like WHO, CDC, government websites, and research papers.
        Provide a short explanation and indicate whether it is True, False, or Misleading.

        Statement: "{statement}"
        
        Response format:
        - Verdict: (True / False / Misleading)
        - Explanation: (Short reason)
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a fact-checking assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.2
            )

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error: {e}"

