# Задание 4 — Генерация текстовых описаний экосистем (GPT-2)

Лабораторная работа: fine-tuning GPT-2 для генерации описаний пищевых цепочек.

---

## Что делает этот проект

1. **prepare_data.py** — превращает таблицу взаимодействий (GloBI) в тексты на английском  
2. **train.py** — дообучает GPT-2 на этих текстах  
3. **generate.py** — генерирует новые описания экосистем  
4. **evaluate.py** — считает **Perplexity** и готовит шаблон для **MOS**

---

## Быстрый старт (Windows)

### Шаг 0. Открой терминал в папке проекта

```powershell
cd "C:\Users\tbnamozova\Desktop\ИТМО\Глубокое обучение\Задание 4"
```

### Шаг 1. Создай виртуальное окружение (рекомендуется)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Если PowerShell ругается на скрипты:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Шаг 2. Установи зависимости

```powershell
pip install -r requirements.txt
```

Первый запуск скачает модель GPT-2 (~500 МБ) из Hugging Face.

### Шаг 3. Запусти весь pipeline одной командой

```powershell
python run_all.py
```

Или по шагам:

```powershell
python src/prepare_data.py
python src/train.py
python src/generate.py
python src/evaluate.py
```

**На CPU обучение может занять 1–3 часа.** Для быстрой проверки:

```powershell
python run_all.py --epochs 1 --max-rows 10000
```

---

## Структура проекта

```
Задание 4/
├── interactions_sample_200k.tsv/   # исходный датасет (GloBI)
│   └── interactions_sample_200k.tsv
├── src/
│   ├── config.py          # все настройки в одном месте
│   ├── prepare_data.py    # TSV → train/val/test.txt
│   ├── train.py           # fine-tuning GPT-2
│   ├── generate.py        # генерация текстов
│   └── evaluate.py        # Perplexity + MOS
├── data/                  # создаётся автоматически
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── models/
│   └── gpt2-ecosystem/    # сохранённая модель после обучения
├── results/
│   ├── generated_descriptions.txt
│   ├── metrics.json
│   ├── training_metrics.json
│   └── mos_evaluation_template.csv
├── run_all.py
├── requirements.txt
└── README.md
```

---

## Подробно: что происходит на каждом этапе

### 1. Подготовка данных (`prepare_data.py`)

**Вход:** TSV с колонками `sourceTaxonName`, `targetTaxonName`, `interactionTypeName`, таксономия.

**Выход:** три файла с текстами, по одному предложению на строку.

Пример преобразования:

| Было в таблице | Стало текстом |
|----------------|---------------|
| Phidippus carneus → preysOn → Diptera | `In this ecosystem, Phidippus carneus preys on Diptera.` |
| Gadus macrocephalus → eats → Snow crab | `In this marine food web, Gadus macrocephalus feeds on Snow crab Chionoecetes opilio.` |

Фильтруются мусорные строки (`same slab`, URL, вирусы без контекста и т.д.).

**Почему английский?** Базовая GPT-2 обучена на английском. Для русского нужна модель вроде `ai-forever/rugpt3small_based_on_gpt2`.

### 2. Обучение (`train.py`)

- Загружается предобученная GPT-2
- На текстах из `data/train.txt` модель учится предсказывать следующее слово
- Сохраняется в `models/gpt2-ecosystem/`

Настройки в `src/config.py`:

| Параметр | Значение | Что значит |
|----------|----------|------------|
| `NUM_TRAIN_EPOCHS` | 3 | Сколько раз модель пройдёт по всем данным |
| `BATCH_SIZE` | 4 | Сколько текстов обрабатывается за раз |
| `LEARNING_RATE` | 5e-5 | Скорость обучения |
| `MAX_SEQ_LENGTH` | 128 | Макс. длина текста в токенах |

### 3. Генерация (`generate.py`)

Загружает обученную модель и продолжает тексты с промптов:

```python
prompt = "In this ecosystem, predators"
# → "In this ecosystem, predators preys on small fish ..."
```

Результаты → `results/generated_descriptions.txt`

**Свой промпт в коде** — отредактируй `GENERATION_PROMPTS` в `src/config.py` или запусти интерактивно:

```powershell
python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from generate import load_generator
gen = load_generator(Path('models/gpt2-ecosystem'))
print(gen('In the forest, wolves', max_new_tokens=60, do_sample=True)[0]['generated_text'])
"
```

### 4. Оценка (`evaluate.py`)

**Perplexity** — автоматически на `data/test.txt`:
- Считается для базовой GPT-2 и для вашей fine-tuned модели
- **Меньше = лучше** (модель меньше «удивляется» текстам)

**MOS (Mean Opinion Score)** — ручная оценка:
1. Открой `results/mos_evaluation_template.csv` в Excel
2. Прочитай каждый сгенерированный текст
3. Поставь оценки 1–5:
   - `coherence_1_to_5` — связность, грамматика
   - `ecological_realism_1_to_5` — правдоподобность экологически
   - `overall_1_to_5` — общее впечатление
4. Сохрани файл и запусти:

```powershell
python src/evaluate.py --compute-mos
```

---

## Пример из методички

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "In this ecosystem, predators"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

После обучения замените `'gpt2'` на `'models/gpt2-ecosystem'` (локальная папка).

---

## GPU / Google Colab

Если нет GPU, используй [Google Colab](https://colab.research.google.com):

1. Загрузи папку проекта на Google Drive
2. Установи зависимости: `!pip install -r requirements.txt`
3. Запусти `!python run_all.py`

Colab даст GPU — обучение займёт ~10–20 минут вместо часов.

---

## Что писать в отчёте

1. **Данные:** GloBI (Global Biotic Interactions), N строк → M текстов после фильтрации  
2. **Модель:** GPT-2 fine-tuning, гиперпараметры из `config.py`  
3. **Perplexity:** base vs fine-tuned из `results/metrics.json`  
4. **MOS:** средняя оценка из CSV  
5. **Примеры генерации:** 5–10 текстов из `generated_descriptions.txt`  
6. **Вывод:** fine-tuning снизил perplexity, тексты стали более «экологичными»

---

## Частые проблемы

| Проблема | Решение |
|----------|---------|
| `python` не найден | Установи Python 3.10+ с python.org, отметь «Add to PATH» |
| Out of memory | Уменьши `BATCH_SIZE` и `MAX_ROWS` в config.py |
| Модель генерирует бессмыслицу | Обучи дольше (`--epochs 5`) или проверь, что train.py завершился |
| Очень долго на CPU | `--epochs 1 --max-rows 5000` для теста |

---

## Связь с GBIF и EcoBase (из задания)

- **GloBI** (ваш TSV) — база биotic interactions, указана в методичке как Predator-Prey Database  
- **GBIF** — данные о видах; мы используем таксономические поля из TSV (class, order, family)  
- **EcoBase** — модели пищевых сетей; имитируем через групповые абзацы `build_food_web_paragraph`

При необходимости можно добавить тексты из GBIF API (поле `description`) — см. [gbif.org/developer/summary](https://www.gbif.org/developer/summary).
