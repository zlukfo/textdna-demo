"""Рендеринг текста с подсветкой."""

import streamlit as st
from core.models import (
    TextAnalysis, SentenceAnalysis,
    ContentType, TextType, EmotionType,
    CONTENT_NAMES, TEXT_TYPE_NAMES, EMOTION_NAMES,
    CONTENT_ICONS, TEXT_TYPE_ICONS, EMOTION_ICONS,
)


CONTENT_COLORS = {
    ContentType.FACT: ("#3498db", "rgba(52, 152, 219, 0.2)"),
    ContentType.EMOTION: ("#e74c3c", "rgba(231, 76, 60, 0.2)"),
    ContentType.ACTION: ("#2ecc71", "rgba(46, 204, 113, 0.2)"),
}


def render_legend():
    """Отображает легенду."""
    cols = st.columns(3)
    with cols[0]:
        st.markdown("📊 **:blue[Факты]**")
    with cols[1]:
        st.markdown("❤️ **:red[Эмоции]**")
    with cols[2]:
        st.markdown("🎯 **:green[Призывы]**")


def render_highlighted_text(analysis: TextAnalysis):
    """Отображает текст с подсветкой."""
    
    if not analysis.sentences:
        st.warning("Нет предложений для анализа.")
        return
    
    render_legend()
    st.divider()
    
    for sentence in analysis.sentences:
        content = sentence.dominant_content
        text_type = sentence.dominant_text_type
        emotion = sentence.dominant_emotion
        
        # Иконки
        c_icon = CONTENT_ICONS[content]
        t_icon = TEXT_TYPE_ICONS[text_type]
        e_icon = EMOTION_ICONS[emotion]
        
        # Цвет
        color_map = {
            ContentType.FACT: "blue",
            ContentType.EMOTION: "red",
            ContentType.ACTION: "green",
        }
        color = color_map[content]
        
        # Отображаем
        st.markdown(
            f"{c_icon} :{color}[{sentence.text}] "
            f"<sup>{t_icon} {e_icon}</sup>",
            unsafe_allow_html=True,
            help=f"Контент: {CONTENT_NAMES[content]}\n"
                 f"Тип: {TEXT_TYPE_NAMES[text_type]}\n"
                 f"Эмоция: {EMOTION_NAMES[emotion]}"
        )


def render_sentence_cards(analysis: TextAnalysis):
    """Отображает карточки предложений."""
    
    if not analysis.sentences:
        st.warning("Нет предложений.")
        return
    
    for i, sentence in enumerate(analysis.sentences, 1):
        content = sentence.dominant_content
        text_type = sentence.dominant_text_type
        emotion = sentence.dominant_emotion
        
        with st.expander(
            f"**#{i}** {CONTENT_ICONS[content]} {TEXT_TYPE_ICONS[text_type]} {EMOTION_ICONS[emotion]} — "
            f"_{sentence.text[:50]}{'...' if len(sentence.text) > 50 else ''}_",
            expanded=False
        ):
            st.markdown(f"**Текст:** {sentence.text}")
            
            st.divider()
            
            # Тип контента
            st.markdown("**📊 Тип контента:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Факт", f"{sentence.content.fact:.0%}")
            col2.metric("Эмоция", f"{sentence.content.emotion:.0%}")
            col3.metric("Призыв", f"{sentence.content.action:.0%}")
            
            # Тип текста
            st.markdown("**📄 Тип текста:**")
            cols = st.columns(5)
            cols[0].metric("Новость", f"{sentence.text_type.news:.0%}")
            cols[1].metric("Реклама", f"{sentence.text_type.advertising:.0%}")
            cols[2].metric("Деловой", f"{sentence.text_type.business:.0%}")
            cols[3].metric("Чат", f"{sentence.text_type.chat:.0%}")
            cols[4].metric("Отзыв", f"{sentence.text_type.review:.0%}")
            
            # Эмоции
            st.markdown("**❤️ Эмоции:**")
            cols = st.columns(4)
            cols[0].metric("😊 Радость", f"{sentence.emotion.joy:.0%}")
            cols[1].metric("😢 Грусть", f"{sentence.emotion.sadness:.0%}")
            cols[2].metric("😠 Злость", f"{sentence.emotion.anger:.0%}")
            cols[3].metric("😨 Страх", f"{sentence.emotion.fear:.0%}")
            
            cols = st.columns(4)
            cols[0].metric("😲 Удивление", f"{sentence.emotion.surprise:.0%}")
            cols[1].metric("🤢 Отвращение", f"{sentence.emotion.disgust:.0%}")
            cols[2].metric("😐 Нейтрально", f"{sentence.emotion.neutral:.0%}")