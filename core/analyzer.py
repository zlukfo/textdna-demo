"""
Анализатор с ИСПРАВЛЕННОЙ загрузкой модели.
"""

import os
import re
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

MODEL_NAME = "cointegrated/rubert-tiny2"

CONTENT_LABELS = ["fact", "emotion", "action"]
TEXT_TYPE_LABELS = ["news", "advertising", "business", "chat", "review"]
EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


# ============================================================
# УЛУЧШЕННАЯ МОДЕЛЬ (та же архитектура что при обучении!)
# ============================================================

class ImprovedMultiHeadClassifier(nn.Module):
    """
    Улучшенная модель - ТОЧНО такая же как при обучении.
    """
    
    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()
        
        from transformers import AutoModel, AutoConfig
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size  # 312 для rubert-tiny2
        
        # Attention pooling - Sequential с индексами 0, 2
        # 0: Linear(312, 156)
        # 1: Tanh()
        # 2: Linear(156, 1)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # attention.0
            nn.Tanh(),                                   # attention.1 (без весов)
            nn.Linear(hidden_size // 2, 1),             # attention.2
        )
        
        # Shared layer - Sequential с индексами 0, 1, 2, 3, 4, 5
        # 0: LayerNorm
        # 1: Dropout (без весов)
        # 2: Linear(312, 512)
        # 3: GELU (без весов)
        # 4: Dropout (без весов)
        # 5: Linear(512, 256)
        # 6: GELU (без весов)
        self.shared_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),       # shared_layer.0
            nn.Dropout(dropout),              # shared_layer.1 (нет весов)
            nn.Linear(hidden_size, 512),     # shared_layer.2
            nn.GELU(),                        # shared_layer.3 (нет весов)
            nn.Dropout(dropout),              # shared_layer.4 (нет весов)
            nn.Linear(512, 256),              # shared_layer.5
            nn.GELU(),                        # shared_layer.6 (нет весов)
        )
        
        # Content head - Sequential с индексами 0, 1, 2, 3
        # 0: Dropout (нет весов)
        # 1: Linear(256, 128)
        # 2: GELU (нет весов)
        # 3: Linear(128, 3)
        self.content_head = nn.Sequential(
            nn.Dropout(dropout),              # content_head.0 (нет весов)
            nn.Linear(256, 128),              # content_head.1
            nn.GELU(),                        # content_head.2 (нет весов)
            nn.Linear(128, len(CONTENT_LABELS))  # content_head.3
        )
        
        # Text type head
        self.text_type_head = nn.Sequential(
            nn.Dropout(dropout),              # text_type_head.0
            nn.Linear(256, 128),              # text_type_head.1
            nn.GELU(),                        # text_type_head.2
            nn.Linear(128, len(TEXT_TYPE_LABELS))  # text_type_head.3
        )
        
        # Emotion head
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),              # emotion_head.0
            nn.Linear(256, 128),              # emotion_head.1
            nn.GELU(),                        # emotion_head.2
            nn.Linear(128, len(EMOTION_LABELS))  # emotion_head.3
        )
    
    def attention_pooling(self, hidden_states, attention_mask):
        """Attention-based pooling."""
        attention_scores = self.attention(hidden_states).squeeze(-1)
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, 
            float('-inf')
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        pooled = torch.bmm(
            attention_weights.unsqueeze(1),
            hidden_states
        ).squeeze(1)
        return pooled
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.attention_pooling(outputs.last_hidden_state, attention_mask)
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
    """
    
    def __init__(self, model_path: Optional[str] = None, prefer_best: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        
        self.model = None
        self.tokenizer = None
        self.use_ml = False
        self.model_version = "unknown"
        self.load_error = None
        
        # Загружаем модель
        self._load_ml_model(model_path, prefer_best)
        
        # Инициализируем эвристики
        self._init_heuristics()
        
        # Логирование
        if self.use_ml:
            print(f"✅ ML-модель активна (версия: {self.model_version})")
        else:
            print(f"⚠️ Используются эвристики: {self.load_error}")
    
    def _load_ml_model(self, model_path: Optional[str] = None, prefer_best: bool = True):
        """Загружает ML-модель."""
        
        # Пути для поиска
        search_paths = []
        
        if model_path:
            search_paths.append(model_path)
        
        if prefer_best:
            search_paths.extend(["./model/best", "./model/final"])
        else:
            search_paths.extend(["./model/final", "./model/best"])
        
        search_paths.extend([
            "model/best", "model/final",
            "/app/model/best", "/app/model/final"
        ])
        
        # Ищем модель
        model_file_path = None
        model_dir = None
        
        for path in search_paths:
            if not os.path.exists(path):
                continue
            
            for filename in ["pytorch_model.bin", "model.bin", "model.pt"]:
                full_path = os.path.join(path, filename)
                if os.path.exists(full_path):
                    size = os.path.getsize(full_path) / (1024 * 1024)
                    if size > 1:
                        model_file_path = full_path
                        model_dir = path
                        print(f"📁 Найдена модель: {full_path} ({size:.1f} MB)")
                        break
            
            if model_file_path:
                break
        
        if not model_file_path:
            self.load_error = "Файл модели не найден"
            return
        
        # Загружаем
        try:
            self._load_from_file(model_file_path, model_dir)
        except Exception as e:
            self.load_error = str(e)
            import traceback
            traceback.print_exc()
    
    def _load_from_file(self, model_file: str, model_dir: str):
        """Загружает модель из файла."""
        
        from transformers import AutoTokenizer
        
        print(f"📂 Загрузка модели из: {model_file}")
        
        # Токенизатор
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            print("✅ Токенизатор загружен")
        except:
            print("⚠️ Используем базовый токенизатор")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Загружаем state_dict
        print("⏳ Загрузка весов...")
        state_dict = torch.load(model_file, map_location=self.device, weights_only=False)
        
        # Выводим ключи для диагностики
        print(f"📊 Ключей в state_dict: {len(state_dict)}")
        print(f"   Первые 5 ключей: {list(state_dict.keys())[:5]}")
        
        # ВСЕГДА создаём Improved модель (так как модель обучена с этой архитектурой)
        print("🔧 Создаём ImprovedMultiHeadClassifier...")
        self.model = ImprovedMultiHeadClassifier(MODEL_NAME)
        self.model_version = "improved"
        
        # Выводим ключи модели для сравнения
        model_keys = list(self.model.state_dict().keys())
        print(f"📊 Ключей в модели: {len(model_keys)}")
        print(f"   Первые 5 ключей: {model_keys[:5]}")
        
        # Загружаем веса (без encoder - он загружается из pretrained)
        # Фильтруем только нужные ключи
        filtered_state_dict = {}
        model_state = self.model.state_dict()
        
        for key in state_dict.keys():
            if key in model_state:
                if state_dict[key].shape == model_state[key].shape:
                    filtered_state_dict[key] = state_dict[key]
                else:
                    print(f"⚠️ Размер не совпадает: {key}")
                    print(f"   state_dict: {state_dict[key].shape}")
                    print(f"   model: {model_state[key].shape}")
            else:
                # Может быть encoder. часть
                if not key.startswith('encoder.'):
                    print(f"⚠️ Ключ не найден в модели: {key}")
        
        print(f"📦 Загружаем {len(filtered_state_dict)} весов...")
        
        # Загружаем
        self.model.load_state_dict(filtered_state_dict, strict=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.use_ml = True
        print(f"✅ Модель загружена успешно!")
    
    def _init_heuristics(self):
        """Инициализация эвристик."""
        
        self.content_markers = {
            'action': ['купи', 'закажи', 'подпиши', 'скачай', 'получи', 'нажми', 'перейди'],
            'fact': ['составля', 'процент', 'данны', 'исследован', 'статистик', 'количеств'],
            'emotion': ['потряса', 'невероят', 'обожа', 'ненавиж', 'ужас', 'восторг']
        }
        
        self.text_type_markers = {
            'news': ['сообща', 'заяви', 'по данным'],
            'advertising': ['скидк', 'акци', 'только сегодня'],
            'business': ['уважаем', 'прошу', 'договор'],
            'chat': ['привет', 'пока', 'лол'],
            'review': ['заказ', 'доставк', 'качеств']
        }
        
        self.emotion_markers = {
            'joy': ['рад', 'счастл', 'восторг'],
            'sadness': ['грустн', 'печаль', 'жаль'],
            'anger': ['бешенств', 'злость', 'возмущ'],
            'fear': ['страш', 'боюсь', 'тревож'],
            'surprise': ['удивл', 'неожидан'],
            'disgust': ['отвратител', 'мерзк'],
            'neutral': []
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        sentences = self._sentence_pattern.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_heuristic(self, sentence: str):
        """Эвристический анализ."""
        from .models import (
            SentenceAnalysis, ContentScores,
            TextTypeScores, EmotionScores
        )
        
        text_lower = sentence.lower()
        
        content_scores = {k: 0.2 for k in CONTENT_LABELS}
        for content_type, markers in self.content_markers.items():
            for marker in markers:
                if marker in text_lower:
                    content_scores[content_type] += 0.3
        
        text_type_scores = {k: 0.15 for k in TEXT_TYPE_LABELS}
        for text_type, markers in self.text_type_markers.items():
            for marker in markers:
                if marker in text_lower:
                    text_type_scores[text_type] += 0.25
        
        emotion_scores = {k: 0.1 for k in EMOTION_LABELS}
        emotion_scores['neutral'] = 0.3
        for emotion, markers in self.emotion_markers.items():
            for marker in markers:
                if marker in text_lower:
                    emotion_scores[emotion] += 0.25
        
        def normalize(scores):
            total = sum(scores.values())
            if total > 0:
                return {k: min(1.0, v / total) for k, v in scores.items()}
            return scores
        
        return SentenceAnalysis(
            text=sentence,
            content=ContentScores(**normalize(content_scores)),
            text_type=TextTypeScores(**normalize(text_type_scores)),
            emotion=EmotionScores(**normalize(emotion_scores)),
            confidence=0.5
        )
    
    def _analyze_ml(self, sentence: str):
        """ML анализ."""
        from .models import (
            SentenceAnalysis, ContentScores,
            TextTypeScores, EmotionScores
        )
        
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
    
    def analyze_sentence(self, sentence: str):
        """Анализирует одно предложение."""
        from .models import (
            SentenceAnalysis, ContentScores,
            TextTypeScores, EmotionScores
        )
        
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
    
    def analyze(self, text: str):
        """Полный анализ текста."""
        from .models import TextAnalysis
        
        sentences = self._split_sentences(text)
        analyzed = [self.analyze_sentence(s) for s in sentences]
        return TextAnalysis(sentences=analyzed, source_text=text)


# ============================================================
# SINGLETON
# ============================================================

@st.cache_resource(ttl=None)
def get_analyzer(model_path: Optional[str] = None, prefer_best: bool = True) -> TextDNAAnalyzer:
    """Возвращает singleton анализатора."""
    return TextDNAAnalyzer(model_path=model_path, prefer_best=prefer_best)

