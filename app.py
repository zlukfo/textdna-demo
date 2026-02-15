"""
TextDNA Analyzer — Мульти-задачный анализ текста

Запуск: streamlit run app.py
"""

import streamlit as st

from core.analyzer import get_analyzer
from core.models import (
    TextAnalysis, Benchmark,
    CONTENT_NAMES, TEXT_TYPE_NAMES, EMOTION_NAMES,
    CONTENT_ICONS, TEXT_TYPE_ICONS, EMOTION_ICONS,
)
from visualization.charts import (
    create_content_radar,
    create_emotion_bar,
    create_text_type_pie,
    create_emotion_pie,
    create_content_flow,
    create_ternary_plot,
)
from visualization.text_renderer import (
    render_highlighted_text,
    render_sentence_cards,
)
from data.examples import EXAMPLES, BENCHMARKS


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

st.set_page_config(
    page_title="TextDNA Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if 'text_input' not in st.session_state:
    st.session_state.text_input = ""


# ============================================================
# CALLBACKS
# ============================================================

def set_example_text(text: str):
    st.session_state.text_input = text
    st.session_state.analysis_result = None


def clear_all():
    st.session_state.text_input = ""
    st.session_state.analysis_result = None


def run_analysis():
    text = st.session_state.text_input
    if text and text.strip():
        analyzer = get_analyzer()
        st.session_state.analysis_result = analyzer.analyze(text)


# ============================================================
# MAIN
# ============================================================

def main():
    # Заголовок
    st.markdown(
        '<h1 style="text-align: center;">🧬 TextDNA Analyzer</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666;">Мульти-задачный анализ текста: контент, стиль и эмоции</p>',
        unsafe_allow_html=True
    )
    
    # Сайдбар
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        st.subheader("📐 Эталон")
        benchmark_key = st.selectbox(
            "Выберите тип:",
            options=list(BENCHMARKS.keys()),
            format_func=lambda x: BENCHMARKS[x].name
        )
        benchmark = BENCHMARKS[benchmark_key]
        st.caption(benchmark.description)
        
        st.divider()
        
        # Информация о модели
        analyzer = get_analyzer()
        if analyzer.use_ml:
            st.success("✅ ML-модель активна")
        else:
            st.warning("⚠️ Эвристики (ML не загружена)")
        
        st.divider()
        
        with st.expander("ℹ️ О проекте"):
            st.markdown("""
            **TextDNA** анализирует текст по трём измерениям:
            
            **📊 Тип контента:**
            - Факт — данные, цифры
            - Эмоция — оценки, чувства
            - Призыв — CTA, действия
            
            **📄 Тип текста:**
            - Новость, Реклама, Деловой, Чат, Отзыв
            
            **❤️ Эмоции:**
            - Радость, Грусть, Злость, Страх
            - Удивление, Отвращение, Нейтрально
            """)
    
    # Примеры
    st.subheader("⚡ Примеры")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, key=f"ex_{name}", use_container_width=True, on_click=set_example_text, args=(text,))
    
    # Ввод
    st.subheader("✍️ Введите текст")
    st.text_area(
        "Текст:",
        height=150,
        placeholder="Вставьте текст для анализа...",
        key="text_input",
        label_visibility="collapsed"
    )
    
    # Кнопки
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("🗑️ Очистить", on_click=clear_all, use_container_width=True)
    with col2:
        st.button(
            "🔬 Анализировать",
            type="primary",
            on_click=run_analysis,
            disabled=not st.session_state.text_input.strip(),
            use_container_width=True
        )
    
    # Результаты
    if st.session_state.analysis_result is not None:
        render_results(st.session_state.analysis_result, benchmark)


def render_results(analysis: TextAnalysis, benchmark: Benchmark):
    """Отображение результатов."""
    
    st.divider()
    st.success(f"✅ Проанализировано **{len(analysis.sentences)}** предложений!")
    
    # ============================================================
    # ОБЩИЕ МЕТРИКИ
    # ============================================================
    
    st.subheader("📈 Общий профиль")
    
    avg_content = analysis.avg_content
    avg_emotion = analysis.avg_emotion
    
    # Тип контента
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "📊 Факты",
        f"{avg_content.fact:.0%}",
        delta=f"{(avg_content.fact - benchmark.fact_target)*100:+.0f}%"
    )
    col2.metric(
        "❤️ Эмоции",
        f"{avg_content.emotion:.0%}",
        delta=f"{(avg_content.emotion - benchmark.emotion_target)*100:+.0f}%"
    )
    col3.metric(
        "🎯 Призывы",
        f"{avg_content.action:.0%}",
        delta=f"{(avg_content.action - benchmark.action_target)*100:+.0f}%"
    )
    col4.metric("📝 Предложений", len(analysis.sentences))
    
    # Доминанты
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dominant_content = avg_content.dominant
        st.markdown(f"**Тип контента:** {CONTENT_ICONS[dominant_content]} {CONTENT_NAMES[dominant_content]}")
    
    with col2:
        # Наиболее частый тип текста
        text_type_dist = analysis.text_type_distribution
        dominant_text_type = max(text_type_dist, key=text_type_dist.get)
        st.markdown(f"**Тип текста:** {TEXT_TYPE_ICONS[dominant_text_type]} {TEXT_TYPE_NAMES[dominant_text_type]}")
    
    with col3:
        dominant_emotion = avg_emotion.dominant
        st.markdown(f"**Доминирующая эмоция:** {EMOTION_ICONS[dominant_emotion]} {EMOTION_NAMES[dominant_emotion]}")
    
    st.divider()
    
    # ============================================================
    # ВКЛАДКИ
    # ============================================================
    
    tabs = st.tabs([
        "🎨 Подсветка",
        "🧬 ДНК-карта",
        "📊 Контент",
        "❤️ Эмоции",
        "📄 Тип текста",
        "📋 Детально"
    ])
    
    with tabs[0]:
        render_highlighted_text(analysis)
    
    with tabs[1]:
        st.plotly_chart(create_ternary_plot(analysis), use_container_width=True)
        st.caption("Каждая точка — предложение. Звезда — центр текста.")
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_content_radar(analysis, benchmark), use_container_width=True)
        with col2:
            st.plotly_chart(create_content_flow(analysis), use_container_width=True)
    
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_emotion_bar(analysis), use_container_width=True)
        with col2:
            st.plotly_chart(create_emotion_pie(analysis), use_container_width=True)
    
    with tabs[4]:
        st.plotly_chart(create_text_type_pie(analysis), use_container_width=True)
        
        # Таблица распределения
        st.subheader("Распределение по типам")
        dist = analysis.text_type_distribution
        for text_type, count in dist.items():
            if count > 0:
                pct = count / len(analysis.sentences)
                st.progress(pct, text=f"{TEXT_TYPE_ICONS[text_type]} {TEXT_TYPE_NAMES[text_type]}: {count} ({pct:.0%})")
    
    with tabs[5]:
        render_sentence_cards(analysis)
    
    # ============================================================
    # РЕКОМЕНДАЦИИ
    # ============================================================
    
    st.divider()
    st.subheader("💡 Рекомендации")
    
    issues = []
    
    avg = analysis.avg_content
    
    if avg.fact < benchmark.fact_target - benchmark.tolerance:
        issues.append(("📊", "Добавьте больше фактов и цифр"))
    if avg.emotion < benchmark.emotion_target - benchmark.tolerance:
        issues.append(("❤️", "Добавьте эмоциональные элементы"))
    if avg.action < benchmark.action_target - benchmark.tolerance:
        issues.append(("🎯", "Усильте призыв к действию"))
    
    # Проверка типа текста
    text_type_dist = analysis.text_type_distribution
    dominant_text_type = max(text_type_dist, key=text_type_dist.get)
    if dominant_text_type != benchmark.expected_text_type:
        issues.append((
            "📄",
            f"Текст больше похож на '{TEXT_TYPE_NAMES[dominant_text_type]}', "
            f"а не на '{TEXT_TYPE_NAMES[benchmark.expected_text_type]}'"
        ))
    
    # Проверка эмоций
    dominant_emotion = avg_emotion.dominant
    if dominant_emotion in [EmotionType.ANGER, EmotionType.DISGUST] and benchmark_key != "review":
        issues.append(("😠", "Обнаружены негативные эмоции, это может оттолкнуть читателя"))
    
    if not issues:
        st.success(f"✅ Отличный баланс для «{benchmark.name}»!")
    else:
        for icon, text in issues:
            st.warning(f"{icon} {text}")
    
    # ============================================================
    # ЭКСПОРТ
    # ============================================================
    
    st.divider()
    st.subheader("💾 Экспорт")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON
        import json
        export_data = {
            "summary": {
                "content": {"fact": avg.fact, "emotion": avg.emotion, "action": avg.action},
                "dominant_content": avg.dominant.value,
                "dominant_text_type": dominant_text_type.value,
                "dominant_emotion": dominant_emotion.value,
            },
            "sentences": [
                {
                    "text": s.text,
                    "content": {"fact": s.content.fact, "emotion": s.content.emotion, "action": s.content.action},
                    "text_type": s.dominant_text_type.value,
                    "emotion": s.dominant_emotion.value,
                }
                for s in analysis.sentences
            ]
        }
        
        st.download_button(
            "📄 JSON",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name="textdna_analysis.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # CSV
        csv_lines = ["text,content_type,text_type,emotion,fact,emotion_score,action"]
        for s in analysis.sentences:
            text_escaped = s.text.replace('"', '""')
            csv_lines.append(
                f'"{text_escaped}",'
                f'{s.dominant_content.value},{s.dominant_text_type.value},{s.dominant_emotion.value},'
                f'{s.content.fact:.3f},{s.content.emotion:.3f},{s.content.action:.3f}'
            )
        
        st.download_button(
            "📊 CSV",
            data="\n".join(csv_lines),
            file_name="textdna_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )


# Импорт EmotionType для проверки
from core.models import EmotionType


if __name__ == "__main__":
    main()