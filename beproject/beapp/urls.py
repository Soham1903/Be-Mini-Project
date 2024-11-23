from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('home/', views.home_view, name='home'), 
    path('analyze-audio/', views.analyze_audio, name='analyze_audio'),
    path("analyze_legal_query/", views.analyze_legal_query, name="analyze_legal_query"),
    path("ocr/", views.ocr_view, name="ocr_view"),
    path('chat/', views.chatbot_view, name='chatbot_view'),
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)