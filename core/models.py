"""Модели данных для мульти-задачного анализатора."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


# ============================================================
# КАТЕГОРИИ
# ============================================================

class ContentType(str, Enum):
    """Тип контента."""
    FACT = "fact"
    EMOTION = "emotion"
    ACTION = "action"


class TextType(str, Enum):
    """Тип текста."""
    NEWS = "news"
    ADVERTISING = "advertising"
    BUSINESS = "business"
    CHAT = "chat"
    REVIEW = "review"


class EmotionType(str, Enum):
    """Тип эмоции."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


# Русские названия
CONTENT_NAMES = {
    ContentType.FACT: "Факт",
    ContentType.EMOTION: "Эмоция",
    ContentType.ACTION: "Призыв",
}

TEXT_TYPE_NAMES = {
    TextType.NEWS: "Новость",
    TextType.ADVERTISING: "Реклама",
    TextType.BUSINESS: "Деловой",
    TextType.CHAT: "Чат",
    TextType.REVIEW: "Отзыв",
}

EMOTION_NAMES = {
    EmotionType.JOY: "Радость",
    EmotionType.SADNESS: "Грусть",
    EmotionType.ANGER: "Злость",
    EmotionType.FEAR: "Страх",
    EmotionType.SURPRISE: "Удивление",
    EmotionType.DISGUST: "Отвращение",
    EmotionType.NEUTRAL: "Нейтрально",
}

EMOTION_ICONS = {
    EmotionType.JOY: "😊",
    EmotionType.SADNESS: "😢",
    EmotionType.ANGER: "😠",
    EmotionType.FEAR: "😨",
    EmotionType.SURPRISE: "😲",
    EmotionType.DISGUST: "🤢",
    EmotionType.NEUTRAL: "😐",
}

CONTENT_ICONS = {
    ContentType.FACT: "📊",
    ContentType.EMOTION: "❤️",
    ContentType.ACTION: "🎯",
}

TEXT_TYPE_ICONS = {
    TextType.NEWS: "📰",
    TextType.ADVERTISING: "📢",
    TextType.BUSINESS: "💼",
    TextType.CHAT: "💬",
    TextType.REVIEW: "⭐",
}


# ============================================================
# МОДЕЛИ ДАННЫХ
# ============================================================

class ContentScores(BaseModel):
    """Оценки типа контента."""
    fact: float = Field(ge=0, le=1)
    emotion: float = Field(ge=0, le=1)
    action: float = Field(ge=0, le=1)
    
    @property
    def dominant(self) -> ContentType:
        scores = {"fact": self.fact, "emotion": self.emotion, "action": self.action}
        return ContentType(max(scores, key=scores.get))


class TextTypeScores(BaseModel):
    """Оценки типа текста."""
    news: float = Field(ge=0, le=1)
    advertising: float = Field(ge=0, le=1)
    business: float = Field(ge=0, le=1)
    chat: float = Field(ge=0, le=1)
    review: float = Field(ge=0, le=1)
    
    @property
    def dominant(self) -> TextType:
        scores = {
            "news": self.news,
            "advertising": self.advertising,
            "business": self.business,
            "chat": self.chat,
            "review": self.review,
        }
        return TextType(max(scores, key=scores.get))


class EmotionScores(BaseModel):
    """Оценки эмоций."""
    joy: float = Field(ge=0, le=1)
    sadness: float = Field(ge=0, le=1)
    anger: float = Field(ge=0, le=1)
    fear: float = Field(ge=0, le=1)
    surprise: float = Field(ge=0, le=1)
    disgust: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    
    @property
    def dominant(self) -> EmotionType:
        scores = {
            "joy": self.joy,
            "sadness": self.sadness,
            "anger": self.anger,
            "fear": self.fear,
            "surprise": self.surprise,
            "disgust": self.disgust,
            "neutral": self.neutral,
        }
        return EmotionType(max(scores, key=scores.get))


class SentenceAnalysis(BaseModel):
    """Результат анализа одного предложения."""
    
    text: str
    content: ContentScores
    text_type: TextTypeScores
    emotion: EmotionScores
    confidence: float = Field(ge=0, le=1, default=1.0)
    
    @property
    def dominant_content(self) -> ContentType:
        return self.content.dominant
    
    @property
    def dominant_text_type(self) -> TextType:
        return self.text_type.dominant
    
    @property
    def dominant_emotion(self) -> EmotionType:
        return self.emotion.dominant


class TextAnalysis(BaseModel):
    """Полный результат анализа текста."""
    
    sentences: List[SentenceAnalysis]
    source_text: str = ""
    
    @property
    def avg_content(self) -> ContentScores:
        if not self.sentences:
            return ContentScores(fact=0, emotion=0, action=0)
        
        return ContentScores(
            fact=sum(s.content.fact for s in self.sentences) / len(self.sentences),
            emotion=sum(s.content.emotion for s in self.sentences) / len(self.sentences),
            action=sum(s.content.action for s in self.sentences) / len(self.sentences),
        )
    
    @property
    def avg_emotion(self) -> EmotionScores:
        if not self.sentences:
            return EmotionScores(
                joy=0, sadness=0, anger=0, fear=0, 
                surprise=0, disgust=0, neutral=0
            )
        
        n = len(self.sentences)
        return EmotionScores(
            joy=sum(s.emotion.joy for s in self.sentences) / n,
            sadness=sum(s.emotion.sadness for s in self.sentences) / n,
            anger=sum(s.emotion.anger for s in self.sentences) / n,
            fear=sum(s.emotion.fear for s in self.sentences) / n,
            surprise=sum(s.emotion.surprise for s in self.sentences) / n,
            disgust=sum(s.emotion.disgust for s in self.sentences) / n,
            neutral=sum(s.emotion.neutral for s in self.sentences) / n,
        )
    
    @property
    def text_type_distribution(self) -> Dict[TextType, int]:
        """Распределение по типам текста."""
        dist = {t: 0 for t in TextType}
        for s in self.sentences:
            dist[s.dominant_text_type] += 1
        return dist
    
    @property
    def emotion_distribution(self) -> Dict[EmotionType, int]:
        """Распределение по эмоциям."""
        dist = {e: 0 for e in EmotionType}
        for s in self.sentences:
            dist[s.dominant_emotion] += 1
        return dist


class Benchmark(BaseModel):
    """Эталонный профиль."""
    
    name: str
    description: str
    fact_target: float = Field(ge=0, le=1)
    emotion_target: float = Field(ge=0, le=1)
    action_target: float = Field(ge=0, le=1)
    expected_text_type: TextType
    tolerance: float = Field(ge=0, le=1, default=0.1)