"""
Мульти-задачный анализатор текста.
Использует обученную модель для классификации.
"""

import os
import re
import json
from typing import List, Dict, Optional

import torch
import torch.nn as nn

from .models import (
    SentenceAnalysis, TextAnalysis,
    ContentScores, TextTypeScores, EmotionScores,
)


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

MODEL_NAME = "cointegrated/rubert-tiny2"

CONTENT_LABELS = ["fact", "emotion", "action"]
TEXT_TYPE_LABELS = ["news", "advertising", "business", "chat", "review"]
EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


# ============================================================
# МОДЕЛЬ
# ============================================================

class MultiHeadClassifier(nn.Module):
    """Модель с тремя классификационными головами."""
    
    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        
        from transformers import AutoModel, AutoConfig
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        
        self.shared_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.content_head = nn.Linear(256, len(CONTENT_LABELS))
        self.text_type_head = nn.Linear(256, len(TEXT_TYPE_LABELS))
        self.emotion_head = nn.Linear(256, len(EMOTION_LABELS))
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        shared_output = self.shared_layer(pooled)
        
        return {
            "content_logits": self.content_head(shared_output),
            "text_type_logits": self.text_type_head(shared_output),
            "emotion_logits": self.emotion_head(shared_output),
        }


# ============================================================
# АНАЛИЗАТОР
# ============================================================

class TextDNAAnalyzer:
    """
    Мульти-задачный анализатор текста.
    
    Классифицирует:
    - Тип контента: fact / emotion / action
    - Тип текста: news / advertising / business / chat / review
    - Эмоция: joy / sadness / anger / fear / surprise / disgust / neutral
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Путь к обученной модели (или None для эвристик)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        
        self.model = None
        self.tokenizer = None
        self.use_ml = False
        
        # Пытаемся загрузить ML-модель
        if model_path:
            self._load_model(model_path)
        else:
            # Ищем модель в стандартных местах
            for path in ["./model/final", "./model/best", "./model"]:
                if os.path.exists(path):
                    self._load_model(path)
                    break
        
        # Инициализируем эвристики как fallback
        self._init_heuristics()
    
    def _load_model(self, model_path: str):
        """Загружает ML-модель."""
        try:
            from transformers import AutoTokenizer
            
            # Проверяем наличие файла модели
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if not os.path.exists(model_file):
                print(f"⚠️ Модель не найдена: {model_file}")
                return
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Загружаем модель
            self.model = MultiHeadClassifier(MODEL_NAME)
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.use_ml = True
            print(f"✅ ML-модель загружена: {model_path}")
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            self.use_ml = False
    
    def _init_heuristics(self):
        """Инициализация эвристических правил (fallback)."""
        
        self.content_markers = {
            'action': [
                'купи', 'закажи', 'подпиши', 'скачай', 'получи',
                'нажми', 'перейди', 'оформи', 'зарегистрируй',
                'успей', 'торопи', 'не упусти', 'начни', 'попробуй'
            ],
            'fact': [
                'составля', 'процент', 'данны', 'исследован',
                'статистик', 'количеств', 'результат', 'показател',
                'уровен', 'равн', 'году', 'рублей', 'долларов'
            ],
            'emotion': [
                'потряса', 'невероят', 'обожа', 'ненавиж',
                'ужас', 'восторг', 'прекрасн', 'отвратител',
                'супер', 'круто', 'класс', 'кошмар'
            ]
        }
        
        self.text_type_markers = {
            'news': ['сообща', 'заяви', 'по данным', 'эксперт', 'аналитик'],
            'advertising': ['скидк', 'акци', 'только сегодня', 'бесплатн', 'бонус'],
            'business': ['уважаем', 'прошу', 'согласован', 'договор', 'отчёт'],
            'chat': ['привет', 'пока', 'лол', 'окей', 'спасибо', 'пожалуйста'],
            'review': ['заказ', 'доставк', 'качеств', 'товар', 'рекоменду']
        }
        
        self.emotion_markers = {
            'joy': ['рад', 'счастл', 'восторг', 'обожа', 'супер', 'класс', 'отличн'],
            'sadness': ['грустн', 'печаль', 'жаль', 'расстро', 'сожален'],
            'anger': ['бешенств', 'злость', 'возмущ', 'безобраз', 'наглост'],
            'fear': ['страш', 'боюсь', 'тревож', 'опаса', 'ужас'],
            'surprise': ['удивл', 'неожидан', 'ничего себе', 'вот это', 'шок'],
            'disgust': ['отвратител', 'мерзк', 'противн', 'фу', 'гадост'],
            'neutral': []
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        sentences = self._sentence_pattern.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_heuristic(self, sentence: str) -> SentenceAnalysis:
        """Эвристический анализ (fallback)."""
        text_lower = sentence.lower()
        
        # Content scores
        content_scores = {k: 0.2 for k in CONTENT_LABELS}
        for content_type, markers in self.content_markers.items():
            for marker in markers:
                if marker in text_lower:
                    content_scores[content_type] += 0.3
        
        # Text type scores
        text_type_scores = {k: 0.15 for k in TEXT_TYPE_LABELS}
        for text_type, markers in self.text_type_markers.items():
            for marker in markers:
                if marker in text_lower:
                    text_type_scores[text_type] += 0.25
        
        # Emotion scores
        emotion_scores = {k: 0.1 for k in EMOTION_LABELS}
        emotion_scores['neutral'] = 0.3
        for emotion, markers in self.emotion_markers.items():
            for marker in markers:
                if marker in text_lower:
                    emotion_scores[emotion] += 0.25
        
        # Пунктуация
        if '!' in sentence:
            emotion_scores['neutral'] -= 0.1
            content_scores['emotion'] += 0.1
        
        # Нормализация
        def normalize(scores):
            total = sum(scores.values())
            if total > 0:
                return {k: min(1.0, v / total) for k, v in scores.items()}
            return scores
        
        content_scores = normalize(content_scores)
        text_type_scores = normalize(text_type_scores)
        emotion_scores = normalize(emotion_scores)
        
        return SentenceAnalysis(
            text=sentence,
            content=ContentScores(**content_scores),
            text_type=TextTypeScores(**text_type_scores),
            emotion=EmotionScores(**emotion_scores),
            confidence=0.5
        )
    
    def _analyze_ml(self, sentence: str) -> SentenceAnalysis:
        """Анализ с использованием ML-модели."""
        
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        content_probs = torch.softmax(outputs['content_logits'], dim=-1)[0].cpu().numpy()
        text_type_probs = torch.softmax(outputs['text_type_logits'], dim=-1)[0].cpu().numpy()
        emotion_probs = torch.softmax(outputs['emotion_logits'], dim=-1)[0].cpu().numpy()
        
        content_scores = {CONTENT_LABELS[i]: float(content_probs[i]) for i in range(len(CONTENT_LABELS))}
        text_type_scores = {TEXT_TYPE_LABELS[i]: float(text_type_probs[i]) for i in range(len(TEXT_TYPE_LABELS))}
        emotion_scores = {EMOTION_LABELS[i]: float(emotion_probs[i]) for i in range(len(EMOTION_LABELS))}
        
        confidence = max(content_probs.max(), text_type_probs.max(), emotion_probs.max())
        
        return SentenceAnalysis(
            text=sentence,
            content=ContentScores(**content_scores),
            text_type=TextTypeScores(**text_type_scores),
            emotion=EmotionScores(**emotion_scores),
            confidence=float(confidence)
        )
    
    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        """Анализирует одно предложение."""
        
        if not sentence or not sentence.strip():
            return SentenceAnalysis(
                text=sentence,
                content=ContentScores(fact=0.33, emotion=0.33, action=0.34),
                text_type=TextTypeScores(news=0.2, advertising=0.2, business=0.2, chat=0.2, review=0.2),
                emotion=EmotionScores(joy=0, sadness=0, anger=0, fear=0, surprise=0, disgust=0, neutral=1.0),
                confidence=0.0
            )
        
        if self.use_ml and self.model is not None:
            try:
                return self._analyze_ml(sentence)
            except Exception as e:
                print(f"⚠️ ML ошибка: {e}")
        
        return self._analyze_heuristic(sentence)
    
    def analyze(self, text: str) -> TextAnalysis:
        """Полный анализ текста."""
        sentences = self._split_sentences(text)
        analyzed = [self.analyze_sentence(s) for s in sentences]
        return TextAnalysis(sentences=analyzed, source_text=text)


# ============================================================
# SINGLETON
# ============================================================

_analyzer_instance = None


def get_analyzer(model_path: Optional[str] = None) -> TextDNAAnalyzer:
    """Возвращает singleton анализатора."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TextDNAAnalyzer(model_path=model_path)
    return _analyzer_instance