from django.db import models

class AudioAnalysis(models.Model):
    # Store the uploaded audio file
    audio_file = models.FileField(upload_to='uploads/audio/', null=True, blank=True)
    
    # Transcription text
    transcription = models.TextField(null=True, blank=True)
    
    # Analysis fields
    speech_rate = models.FloatField(null=True, blank=True)
    most_used_words = models.JSONField(null=True, blank=True)  # store as JSON data
    filler_word_count = models.IntegerField(null=True, blank=True)
    vocabulary_richness = models.FloatField(null=True, blank=True)
    avg_pitch = models.FloatField(null=True, blank=True)
    avg_loudness = models.FloatField(null=True, blank=True)
    loudness_variation = models.FloatField(null=True, blank=True)
    emotion_analysis = models.CharField(max_length=100, null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)



    def __str__(self):
        return f"Audio Analysis {self.id}"

class LegalQueryAnalysis(models.Model):
    # Store the uploaded audio file
    audio_file = models.FileField(upload_to='uploads/legal_queries/', null=True, blank=True)
    
    # Transcription text
    transcription = models.TextField(null=True, blank=True)
    
    # Fact-checking results (one entry per sentence)
    fact_check_results = models.JSONField(null=True, blank=True)  # store as JSON data
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Legal Query Analysis {self.id}"
    
class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/images/')