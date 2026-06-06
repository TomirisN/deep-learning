# Лабораторная работа 3 — Agent AI

Агентная система для поиска научной литературы и генерации мини-обзоров с сравнением режимов **baseline**, **agent** и **agent + evaluator**.

## Что реализовано

| Компонент | Файл | Описание |
|-----------|------|----------|
| State | `state.py` | Состояние агента: история, источники, trace |
| Tools | `tools.py` | OpenAlex API, Wikipedia API, восстановление abstract |
| Pipeline | `pipeline.py` | `run_baseline`, `run_agent`, `save_trace` |
| Evaluator | `evaluator.py` | Оценка ответа по rubric (0–5) |
| LLM | `llm_client.py` | Клиент Ollama |
| Experiments | `experiments.py` | Прогон всех конфигураций, CSV, графики |

### Конфигурации экспериментов

1. **baseline** — один промпт + Wikipedia
2. **agent** — Wikipedia + OpenAlex + structured notes + trace
3. **agent+evaluator** — agent с уточнением ответа по feedback evaluator
4. **agent_top{3,5,8}** — вариация числа источников
5. **agent_steps{4,6,8}** — вариация max_steps

### Темы (8 штук)

- Agentic AI for customer support
- Graph RAG for enterprise knowledge systems
- LLM evaluation and process-aware metrics
- Tool-using language models in scientific search
- Retrieval-augmented generation in medicine
- Planning and reflection in LLM agents
- Human-in-the-loop AI systems
- Knowledge graphs for procedural reasoning

---

## Инструкция по запуску

### 1. Установка Ollama

1. Скачайте и установите [Ollama](https://ollama.com).
2. В терминале запустите сервер (обычно стартует автоматически):
   ```bash
   ollama serve
   ```
3. Скачайте модель (рекомендуется):
   ```bash
   ollama pull llama3.2
   ```
   Альтернативы: `mistral`, `llama3.1`, `qwen2.5:7b`.

### 2. Установка Python-зависимостей

```bash
cd "C:\Users\tbnamozova\Desktop\ИТМО\Глубокое обучение\лаба 3"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Быстрый тест (2 темы, ~10–20 мин)

```bash
python main.py --quick
```

### 4. Полный прогон (8 тем, все конфигурации, ~1–3 часа)

```bash
python main.py
```

### 5. Дополнительные параметры

```bash
# Другая модель
python main.py --model mistral

# Удалённый Ollama
python main.py --base-url http://192.168.1.10:11434

# Без графиков
python main.py --no-plots
```

---

## Результаты

После прогона в папке `outputs/`:

| Файл | Содержимое |
|------|------------|
| `results.csv` | Все записи по темам и режимам |
| `summary.csv` | Средние метрики по режимам |
| `summary.json` | То же в JSON |
| `quality_comparison.png` | График качества (correctness, groundedness, …) |
| `performance_comparison.png` | График latency и числа шагов |
| `traces/<mode>/*.json` | Trace агента для каждой темы |

### Метрики

- **correctness** — корректность ответа (evaluator, 0–5)
- **groundedness** — опора на источники
- **completeness** — полнота обзора
- **coverage** — наличие обязательных разделов
- **rubric** — среднее по 5 критериям
- **n_steps** — число шагов в trace
- **latency** — время выполнения (сек)

---

## Проверка отдельных компонентов

```bash
# Только API (без LLM)
python -c "from tools import search_openalex, search_wikipedia; print(search_wikipedia('Graph RAG')[:200]); print(len(search_openalex('LLM agents', 3)))"
```

---

## Требования к отчёту (из задания)

1. Сравнить **baseline** vs **agent** vs **agent+evaluator** по rubric и latency.
2. Проанализировать trace: какие tools вызывались, где grounding слабее.
3. Объяснить, почему agent даёт более grounded ответы при большей latency.
4. Приложить `summary.csv`, графики и 1–2 примера trace.

---

## Возможные проблемы

| Проблема | Решение |
|----------|---------|
| `Connection refused` к Ollama | Запустите `ollama serve`, проверьте порт 11434 |
| Модель не найдена | `ollama pull llama3.2` |
| Timeout на LLM | Уменьшите темы: `--quick` или смените модель на меньшую |
| OpenAlex/Wikipedia недоступны | Проверьте интернет; API бесплатные, без ключа |
| JSON от evaluator не парсится | Код автоматически извлекает JSON из markdown-ответа |
