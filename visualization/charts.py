"""Графики и визуализации."""

import plotly.graph_objects as go
import plotly.express as px

from core.models import (
    TextAnalysis, Benchmark,
    ContentType, TextType, EmotionType,
    CONTENT_NAMES, TEXT_TYPE_NAMES, EMOTION_NAMES,
)


# Цвета
CONTENT_COLORS = {
    ContentType.FACT: "#3498db",
    ContentType.EMOTION: "#e74c3c",
    ContentType.ACTION: "#2ecc71",
}

EMOTION_COLORS = {
    EmotionType.JOY: "#f1c40f",
    EmotionType.SADNESS: "#3498db",
    EmotionType.ANGER: "#e74c3c",
    EmotionType.FEAR: "#9b59b6",
    EmotionType.SURPRISE: "#e67e22",
    EmotionType.DISGUST: "#27ae60",
    EmotionType.NEUTRAL: "#95a5a6",
}

TEXT_TYPE_COLORS = {
    TextType.NEWS: "#3498db",
    TextType.ADVERTISING: "#e74c3c",
    TextType.BUSINESS: "#2c3e50",
    TextType.CHAT: "#9b59b6",
    TextType.REVIEW: "#f39c12",
}


def create_content_radar(analysis: TextAnalysis, benchmark: Benchmark = None) -> go.Figure:
    """Радар для типов контента."""
    
    categories = ['Факты', 'Эмоции', 'Призывы']
    avg = analysis.avg_content
    
    values = [avg.fact, avg.emotion, avg.action]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color='#3498db', width=3),
        name='Ваш текст'
    ))
    
    if benchmark:
        bench_values = [benchmark.fact_target, benchmark.emotion_target, benchmark.action_target]
        bench_values_closed = bench_values + [bench_values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=bench_values_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(46, 204, 113, 0.2)',
            line=dict(color='#2ecc71', width=2, dash='dash'),
            name=f'Эталон: {benchmark.name}'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%')),
        showlegend=True,
        title="📊 Профиль контента",
        height=400
    )
    
    return fig


def create_emotion_bar(analysis: TextAnalysis) -> go.Figure:
    """Столбчатая диаграмма эмоций."""
    
    avg = analysis.avg_emotion
    
    emotions = list(EMOTION_NAMES.keys())
    values = [
        avg.joy, avg.sadness, avg.anger, 
        avg.fear, avg.surprise, avg.disgust, avg.neutral
    ]
    names = [EMOTION_NAMES[e] for e in emotions]
    colors = [EMOTION_COLORS[e] for e in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f'{v:.0%}' for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="❤️ Распределение эмоций",
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        height=350
    )
    
    return fig


def create_text_type_pie(analysis: TextAnalysis) -> go.Figure:
    """Круговая диаграмма типов текста."""
    
    dist = analysis.text_type_distribution
    
    labels = []
    values = []
    colors = []
    
    for text_type, count in dist.items():
        if count > 0:
            labels.append(TEXT_TYPE_NAMES[text_type])
            values.append(count)
            colors.append(TEXT_TYPE_COLORS[text_type])
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='percent+label',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="📄 Типы текста",
        height=350,
        showlegend=False
    )
    
    return fig


def create_emotion_pie(analysis: TextAnalysis) -> go.Figure:
    """Круговая диаграмма эмоций по предложениям."""
    
    dist = analysis.emotion_distribution
    
    labels = []
    values = []
    colors = []
    
    for emotion, count in dist.items():
        if count > 0:
            labels.append(EMOTION_NAMES[emotion])
            values.append(count)
            colors.append(EMOTION_COLORS[emotion])
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title="😊 Эмоции в предложениях",
        height=350,
        showlegend=False
    )
    
    return fig


def create_content_flow(analysis: TextAnalysis) -> go.Figure:
    """График динамики контента."""
    
    if not analysis.sentences:
        return go.Figure()
    
    n = len(analysis.sentences)
    x = list(range(1, n + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=[s.content.fact for s in analysis.sentences],
        mode='lines+markers', name='Факты',
        line=dict(color='#3498db', width=2),
        fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=[s.content.emotion for s in analysis.sentences],
        mode='lines+markers', name='Эмоции',
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=[s.content.action for s in analysis.sentences],
        mode='lines+markers', name='Призывы',
        line=dict(color='#2ecc71', width=2)
    ))
    
    fig.update_layout(
        title="📈 Динамика по предложениям",
        xaxis_title="Предложение",
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        height=350,
        hovermode='x unified'
    )
    
    return fig


def create_ternary_plot(analysis: TextAnalysis) -> go.Figure:
    """Тернарная диаграмма."""
    
    if not analysis.sentences:
        return go.Figure()
    
    facts = [s.content.fact for s in analysis.sentences]
    emotions = [s.content.emotion for s in analysis.sentences]
    actions = [s.content.action for s in analysis.sentences]
    texts = [s.text[:40] + "..." if len(s.text) > 40 else s.text for s in analysis.sentences]
    colors = [CONTENT_COLORS[s.dominant_content] for s in analysis.sentences]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterternary(
        a=facts, b=emotions, c=actions,
        mode='markers',
        marker=dict(size=12, color=colors, line=dict(width=1, color='white'), opacity=0.8),
        text=texts,
        hovertemplate="<b>%{text}</b><br>Факты: %{a:.0%}<br>Эмоции: %{b:.0%}<br>Призывы: %{c:.0%}<extra></extra>"
    ))
    
    # Центр масс
    avg = analysis.avg_content
    fig.add_trace(go.Scatterternary(
        a=[avg.fact], b=[avg.emotion], c=[avg.action],
        mode='markers',
        marker=dict(size=20, color='gold', symbol='star', line=dict(width=2, color='black')),
        name='Центр'
    ))
    
    fig.update_layout(
        title="🧬 ДНК текста",
        ternary=dict(
            aaxis=dict(title="Факты", color="#3498db"),
            baxis=dict(title="Эмоции", color="#e74c3c"),
            caxis=dict(title="Призывы", color="#2ecc71"),
        ),
        height=500
    )
    
    return fig