from django.shortcuts import render
from django.http import JsonResponse
import speech_recognition as sr
import librosa
import nltk
from nltk.corpus import stopwords
from collections import Counter
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deepmultilingualpunctuation import PunctuationModel
from pydub import AudioSegment
from .models import AudioAnalysis, LegalQueryAnalysis
from pydub.silence import split_on_silence
import numpy as np
import os
import json
from PIL import Image, ImageEnhance
import pytesseract
from .forms import ImageUploadForm
from .utils.speaker_identification import SpeakerIdentification
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import noisereduce as nr
from pydub import AudioSegment
import whisper
from moviepy import VideoFileClip
import soundfile as sf

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
 # your path may be different
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from google.cloud import speech_v1p1beta1 as speech
punctuation_model = PunctuationModel()

def home_view(request):
    return render(request, 'home.html')

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except (sr.UnknownValueError, sr.RequestError):
            return None


def analyze_speech_rate(text):
    word_count = len(text.split())
    duration = len(text) / 160  # Average speaking rate
    return word_count / duration

def analyze_most_used_words(text):
    words = [word.lower() for word in text.split() if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return Counter(words).most_common(10)

def filler_word_count(text):
    fillers = ['uh', 'um', 'you know', 'like']
    return sum(text.lower().count(filler) for filler in fillers)

def vocabulary_richness(text):
    words = text.split()
    unique_words = set(words)
    return len(unique_words) / len(words)

def analyze_voice_modulation(file_path):
    y, sr = librosa.load(file_path, sr=None)
    pitch = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    avg_pitch = np.mean(pitch).item()
    loudness = librosa.feature.rms(y=y)
    avg_loudness = np.mean(loudness).item()
    loudness_variation = np.std(loudness).item()
    return avg_pitch, avg_loudness, loudness_variation

def analyze_emotion(text):
    # Perform emotion analysis
    result = emotion_analyzer(text)
    
    # Return the emotion with the highest score
    emotion = result[0]['label']
    confidence = result[0]['score']
    
    return {"emotion": emotion, "confidence": confidence}

def analyze_audio(request):
    if request.method == 'POST':
        audio_file = request.FILES['audio_file']
        
        # Save audio file
        analysis = AudioAnalysis(audio_file=audio_file)
        analysis.save()
        audio_path = analysis.audio_file.path
        
        # Perform transcription and analyses
        text = transcribe_audio(audio_path)
        if text is None:
            analysis.delete()  # Remove the model entry if transcription fails
            return JsonResponse({"error": "Failed to transcribe audio"})

        # Run all analyses
        analysis.transcription = text
        analysis.speech_rate = round(analyze_speech_rate(text), 2)
        analysis.most_used_words = analyze_most_used_words(text)  # Store JSON-compatible data
        analysis.filler_word_count = filler_word_count(text)
        analysis.vocabulary_richness = round(vocabulary_richness(text) * 100, 2)
        
        # Voice modulation analysis
        avg_pitch, avg_loudness, loudness_variation = analyze_voice_modulation(audio_path)
        analysis.avg_pitch = round(avg_pitch, 2)
        analysis.avg_loudness = round(avg_loudness * 1000, 2)
        analysis.loudness_variation = round(loudness_variation * 1000, 2)
        
        # Emotion analysis
        analysis.emotion_analysis = analyze_emotion(text)["emotion"]
        
        # Save the analysis results
        analysis.save()

        # Prepare the response data
        results = {
            "transcription": text,
            "speech_rate": f"{analysis.speech_rate} per min",
            "most_used_words": analysis.most_used_words,
            "filler_word_count": analysis.filler_word_count,
            "vocabulary_richness": f"{analysis.vocabulary_richness}%",
            "avg_pitch": analysis.avg_pitch,
            "avg_loudness": f"{analysis.avg_loudness} dB",
            "loudness_variation": f"{analysis.loudness_variation} dB",
            "emotion_analysis": analysis.emotion_analysis
        }

        # Clean up the saved audio file locally
        os.remove(audio_path)

        return JsonResponse(results, safe=False)

    return render(request, 'audio_upload.html')

def load_data():
    with open('D:\Be mini-project\datasets\constitution_qa.json', encoding="utf8") as f1, \
         open('D:\Be mini-project\datasets\crpc_qa.json', encoding="utf8") as f2, \
         open('D:\Be mini-project\datasets\ipc_qa.json', encoding="utf8") as f3:
        data = json.load(f1) + json.load(f2) + json.load(f3)
    return data

data = load_data()
questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

def is_subset_of_answer(user_query, answer):
    return user_query.lower() in answer.lower()

def find_best_match(user_query):
    sentences = [sentence.strip() for sentence in user_query.split('.') if sentence.strip()]
    results = []

    for sentence in sentences:
        query_vector = vectorizer.transform([sentence])
        similarities = cosine_similarity(query_vector, question_vectors).flatten()
        best_match_index = similarities.argmax()
        best_match_score = similarities[best_match_index]

        if best_match_score > 0.5:
            best_match_question = questions[best_match_index]
            best_match_answer = answers[best_match_index]
            is_match = is_subset_of_answer(sentence, best_match_answer)

            results.append({
                "Sentence": sentence,
                "Best Match Question": best_match_question,
                "Answer": best_match_answer,
                "Similarity Score": best_match_score,
                "Result": is_match
            })
        else:
            results.append({
                "Sentence": sentence,
                "message": "No relevant legal information found for this sentence."
            })

    return results

def restore_punctuation(text):
    return punctuation_model.restore_punctuation(text)


# Django view to handle the POST request
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def analyze_legal_query(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')

        if not audio_file:
            return JsonResponse({"error": "No audio file provided."}, status=400)

        # Save uploaded audio file to model
        legal_query = LegalQueryAnalysis(audio_file=audio_file)
        legal_query.save()
        audio_path = legal_query.audio_file.path

        # Perform transcription
        text = transcribe_audio(audio_path)
        if text is None:
            legal_query.delete()  # Remove entry if transcription fails
            os.remove(audio_path)
            return JsonResponse({"error": "Failed to transcribe audio"}, status=400)

        # Run fact-checking based on the transcription
        punctuated_text = restore_punctuation(text)
        sentences = nltk.sent_tokenize(punctuated_text)
        all_fact_check_results = []

        for i, sentence in enumerate(sentences, 1):
            fact_check_results = find_best_match(sentence)
            all_fact_check_results.append(fact_check_results)

        # Store transcription and fact-check results in model
        legal_query.transcription = punctuated_text
        legal_query.fact_check_results = all_fact_check_results
        legal_query.save()

        # Clean up the saved audio file locally
        os.remove(audio_path)

        return JsonResponse(all_fact_check_results, safe=False)

    return render(request, 'audio_query.html')
   
def ocr_view(request):
    text = None
    all_fact_check_results = []
    image_url = None  # Variable to store the image URL
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image to a model and get its URL
            uploaded_image = form.save()  # Assuming your form saves the image to a model
            image_url = uploaded_image.image.url  # Get the image URL
            
            image = Image.open(request.FILES['image'])
            
            # OCR and fact-check processing
            text = pytesseract.image_to_string(image, lang='eng')
            if text:
                sentences = nltk.sent_tokenize(text)
                
                for sentence in sentences:
                    fact_check_results = find_best_match(sentence)
                    all_fact_check_results.append({'sentence': sentence, 'facts': fact_check_results})
            else:
                return JsonResponse({"error": "Failed to transcribe text"}, status=400)
    else:
        form = ImageUploadForm()
        
    return render(request, 'ocr_view.html', {
        'form': form,
        'text': text,
        'all_fact_check_results': all_fact_check_results,
        'image_url': image_url,  # Pass the image URL to the template
    })

import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse

# Configure your Google API Key
GOOGLE_API_KEY = 'AIzaSyBb050DVRILhyPuzJVmR4nvvt6qBljm3Qg'
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        # Read JSON body
        data = json.loads(request.body)
        user_message = data.get('message')
        
        if user_message.lower() == "exit":
            return JsonResponse({'response': "Goodbye!"})

        # Generate the chatbot's response
        response = chat.send_message(user_message, stream=True)
        chatbot_response = ''.join(chunk.text for chunk in response if chunk.text)

        return JsonResponse({'response': chatbot_response})

    return render(request, 'chatbot.html')

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
@csrf_exempt
def process_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        # Save the uploaded file temporarily
        video_path = os.path.join("media", video_file.name)
        with open(video_path, "wb+") as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        try:
            # Process the video using SpeakerIdentification class
            processor = SpeakerIdentification(video_path)
            transcripts = processor.process_video(n_speakers=2)

            # ✅ Prepare statements for batch fact-checking
            statements = [entry["text"] for entry in transcripts]
            print(statements)
            fact_check_results = batch_fact_check(statements)
            print(fact_check_results)

            # ✅ Return Transcriptions & Fact Check Results
            return JsonResponse({
                "status": "success",
                "transcripts": transcripts,
                "facts": fact_check_results
            })

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return render(request, "upload.html")

import openai
import json
openai.api_key = os.getenv("OPENAI_API_KEY")# No code was selected, so I will provide a general improvement to the code.

# Add error handling to the batch_fact_check function
def batch_fact_check(statements):
    """Fact-check each statement individually using Gemini."""
    results = []

    for statement in statements:
        try:
            # ✅ Define prompt for a single statement
            prompt = f"""
            Fact-check the following statement:
            
            "{statement}"
            
            Return a JSON object with:
            {{
                "statement": "{statement}",
                "verdict": "True / False / Misleading",
                "explanation": "Short reason"
            }}
            """

            # ✅ Send request to Gemini
            response = genai.GenerativeModel("gemini-pro").generate_content(prompt)

            # ✅ Extract and parse the response
            fact_check_text = response.text  # Gemini returns plain text
            fact_check_result = json.loads(fact_check_text)  # Convert to dictionary

            # ✅ Append to results
            results.append(fact_check_result)

        except json.JSONDecodeError as e:
            results.append({
                "statement": statement,
                "verdict": "Unknown",
                "explanation": f"Failed to parse JSON response: {str(e)}"
            })
        except Exception as e:
            results.append({
                "statement": statement,
                "verdict": "Unknown",
                "explanation": f"An error occurred: {str(e)}"
            })

    # ✅ Debug Output
    print(json.dumps(results, indent=2))  

    return results
