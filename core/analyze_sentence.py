from razdel import sentenize
import torch
from transformers import pipeline
from dataclasses import dataclass, field

# === Разделяет исходный текст на предложения ===
def get_sentences(text: str) -> list[str]:
    return [sentence.text for sentence in sentenize(text)]
# ===============================================

# === Распознает главную эмоцию в предложении ===
emotion_classifier = pipeline(
    "text-classification",
    model="cointegrated/rubert-tiny2-cedr-emotion-detection", 
    device=0 if torch.cuda.is_available() else -1  # Используем GPU, если доступна
)
def get_sentence_emotion(sentence: str) -> dict:
    return emotion_classifier(sentence)[0]
# ===============================================

# === Общие методы, полезные для распознавания эмоций и фактов
@dataclass
class Mixin:
    def _collect_phrase_tokens(self, start_token):
        """
        Собирает связанные токены для start_token
        """
        phrase_tokens = [start_token]
        visited = set([start_token.id])
        
        # Рекурсивно собираем все зависимые токены
        stack = [start_token]
        while stack:
            current = stack.pop()
            
            for token in self.doc.tokens:
                if token.id not in visited and token.head_id == current.id:
                    if token.rel in ['amod', 'nummod', 'det', 'compound', 'case', 
                                    'flat:name', 'nmod', 'appos', 'flat', 'fixed', 'advmod']:
                        phrase_tokens.append(token)
                        visited.add(token.id)
                        stack.append(token)
        
        # Сортируем по позиции в предложении
        phrase_tokens.sort(key=lambda x: x.id)
        return phrase_tokens
# ===============================================


from natasha.doc import DocToken
from natasha import (Segmenter, NewsMorphTagger, NewsEmbedding, Doc, NewsSyntaxParser, NewsNERTagger)
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

# === Распознаем побуждение к действию в предложении ===
@dataclass
class ActionFeature(Mixin):
    text: str                                               # исходный текст
    action: list[DocToken] = field(default_factory=list)    # токены побуждающие к действию
    triggers: list[DocToken] = field(default_factory=list)  # токены-триггеры, которые часто присутсвуют в побуждающих предложениях 
    doc: Doc = None

    def __init__(self, text):
        self.text = text
        self.action = []
        self.triggers = []
        self.doc = Doc(self.text)
        self.doc.segment(segmenter)
        self.doc.tag_morph(morph_tagger)
        self.doc.parse_syntax(syntax_parser)
        self._find_features()
    
    def _find_features(self):
        lexical_triggers = [
                "давай", "пусть", "пускай", "прошу", "требую",
                "советую", "разрешите", "можно", "нужно", "следует", "обязан", "пожалуйста",
                "немедленно", "сейчас", "только", "быстро", "теперь", "немедля", 
                "предлагаю", "рекомендуем", "необходимо", "просим", "призываем",
                "настойчиво", "обращаемся", "требуется", "советуют"
            ]

        for token in self.doc.tokens:
            # Правила для определения глаголов, побуждающих к действию
            # (правило для повелительных глаголов) or (правило для инфинитивов)
            if (token.feats and token.feats.get("Mood") == "Imp") or (token.pos == 'VERB' and token.feats.get("VerbForm") == "Inf"):
                action_tokens = self._collect_phrase_tokens(token)
                self.action.append(action_tokens)

            # Поиск по списку токенов, которые часто пристутсвуют в предложениях побуждающих к действию
            if token.text.lower() in lexical_triggers:
                triggers_tokens = self._collect_phrase_tokens(token)
                self.triggers.append(triggers_tokens)
        return
    
    def get_action_features(self):
        result = {
            'actions': [{'text':_[0].text, 'rel': _[0].rel, 'pos': _[0].pos, 'feats': _[0].feats } for _ in self.action], 
            'triggers': [{'text':_[0].text, 'rel': _[0].rel, 'pos': _[0].pos, 'feats': _[0].feats } for _ in self.triggers]
        }
        return result


    
if __name__ == "__main__":
    text = 'необходимо вставать гулять'
    action = ActionFeature(text)
    print (action.get_action_features())
    #for i in action.action:
    #    for j in i:
    #        print(j.text)


