"""
Лабораторная работа 3 — Agent AI для научных мини-обзоров.

Запуск:
  python main.py                  # полный прогон (8 тем, все конфигурации)
  python main.py --quick          # быстрый тест (2 темы, 3 конфигурации)
  python main.py --model llama3.2 # указать модель Ollama
"""

import argparse
import sys

from config import DEFAULT_MODEL, TOPICS
from experiments import plot_results, run_all_experiments
from llm_client import make_ollama_llm


def check_ollama(llm_call) -> None:
    try:
        response = llm_call("Ответь одним словом: OK")
        print(f"Ollama доступна. Тестовый ответ: {response[:50]}")
    except Exception as exc:
        print(
            "Ошибка подключения к Ollama.\n"
            "1. Установите Ollama: https://ollama.com\n"
            "2. Запустите: ollama serve\n"
            "3. Скачайте модель: ollama pull llama3.2\n"
            f"Детали: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 3 — Agent AI experiments")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Имя модели Ollama (по умолчанию: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="URL Ollama API (по умолчанию: http://localhost:11434)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый прогон: 2 темы, без конфигураций 4–5",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не строить графики",
    )
    args = parser.parse_args()

    print(f"Модель: {args.model}")
    print(f"Тем для прогона: {2 if args.quick else len(TOPICS)}")
    llm_call = make_ollama_llm(model=args.model, base_url=args.base_url)
    check_ollama(llm_call)

    df = run_all_experiments(llm_call, quick=args.quick)

    print("\n=== Сводная таблица (средние по режимам) ===")
    summary = df.groupby("mode")[
        [
            "correctness",
            "groundedness",
            "completeness",
            "coverage",
            "rubric",
            "n_steps",
            "latency",
        ]
    ].mean()
    print(summary.round(3).to_string())

    if not args.no_plots:
        plot_results(df)

    print("\nГотово. Результаты: outputs/results.csv, outputs/summary.csv")


if __name__ == "__main__":
    main()
