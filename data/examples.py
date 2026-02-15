"""Примеры и эталоны."""

from core.models import Benchmark, TextType


EXAMPLES = {
    "📢 Реклама": """Невероятная распродажа только сегодня! Скидки до 70% на всё! Тысячи довольных клиентов уже оценили качество. Не упустите свой шанс! Оформляйте заказ прямо сейчас!""",
    
    "📰 Новость": """Центральный банк принял решение о снижении ключевой ставки на 0.5 процентных пункта. Новая ставка составит 7.5% годовых. Эксперты прогнозируют рост ипотечного кредитования на 10-12%.""",
    
    "💼 Деловое": """Добрый день, уважаемые коллеги. По итогам анализа квартальной отчётности выручка выросла на 15%. Прошу подготовить детализированный отчёт к понедельнику. Встреча состоится в 14:00.""",
    
    "💬 Отзыв": """Это худший опыт покупки в моей жизни! Заказ шёл две недели вместо трёх дней. Упаковка разорвана, товар повреждён. Я в бешенстве! Никому не рекомендую!""",
    
    "🎯 Мотивация": """Ты способен на большее! Каждый великий путь начинается с первого шага. Страх — это иллюзия. Прорывайся! Запишись на вебинар и измени свою жизнь!""",
}


BENCHMARKS = {
    "advertising": Benchmark(
        name="Реклама",
        description="Эмоции + призывы к действию",
        fact_target=0.20,
        emotion_target=0.45,
        action_target=0.35,
        expected_text_type=TextType.ADVERTISING,
        tolerance=0.12
    ),
    "news": Benchmark(
        name="Новости",
        description="Максимум фактов, минимум эмоций",
        fact_target=0.75,
        emotion_target=0.10,
        action_target=0.15,
        expected_text_type=TextType.NEWS,
        tolerance=0.10
    ),
    "business": Benchmark(
        name="Деловая переписка",
        description="Факты + действия",
        fact_target=0.50,
        emotion_target=0.15,
        action_target=0.35,
        expected_text_type=TextType.BUSINESS,
        tolerance=0.10
    ),
    "review": Benchmark(
        name="Отзывы",
        description="Эмоции + факты",
        fact_target=0.35,
        emotion_target=0.50,
        action_target=0.15,
        expected_text_type=TextType.REVIEW,
        tolerance=0.12
    ),
}