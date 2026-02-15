"""Core модуль."""

from .models import (
    SentenceAnalysis, TextAnalysis, Benchmark,
    ContentScores, TextTypeScores, EmotionScores,
    ContentType, TextType, EmotionType,
    CONTENT_NAMES, TEXT_TYPE_NAMES, EMOTION_NAMES,
    CONTENT_ICONS, TEXT_TYPE_ICONS, EMOTION_ICONS,
)
from .analyzer import get_analyzer, TextDNAAnalyzer

__all__ = [
    "SentenceAnalysis", "TextAnalysis", "Benchmark",
    "ContentScores", "TextTypeScores", "EmotionScores",
    "ContentType", "TextType", "EmotionType",
    "CONTENT_NAMES", "TEXT_TYPE_NAMES", "EMOTION_NAMES",
    "CONTENT_ICONS", "TEXT_TYPE_ICONS", "EMOTION_ICONS",
    "get_analyzer", "TextDNAAnalyzer",
]