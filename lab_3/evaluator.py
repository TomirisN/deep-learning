from typing import Callable, List

from llm_client import extract_json

EVAL_PROMPT = """
Оцените мини-обзор по шкале от 0 до 5 по критериям:
1. correctness — фактическая корректность и соответствие теме
2. groundedness — опора на предоставленные источники, отсутствие галлюцинаций
3. completeness — полнота раскрытия темы
4. coverage_of_required_fields — наличие разделов: определение, ключевые работы, идеи, ограничения, перспективы
5. source_consistency — согласованность с abstract источников

Верните ТОЛЬКО JSON без markdown:
{
  "correctness": int,
  "groundedness": int,
  "completeness": int,
  "coverage_of_required_fields": int,
  "source_consistency": int,
  "comment": "..."
}
"""


def evaluate_answer(answer: str, notes: list, llm_call: Callable[[str], str]) -> dict:
    import json

    prompt = (
        EVAL_PROMPT
        + "\n\nОтвет:\n"
        + answer
        + "\n\nИсточники:\n"
        + json.dumps(notes, ensure_ascii=False, indent=2)
    )
    raw = llm_call(prompt)
    result = extract_json(raw)

    defaults = {
        "correctness": 0,
        "groundedness": 0,
        "completeness": 0,
        "coverage_of_required_fields": 0,
        "source_consistency": 0,
        "comment": "",
    }
    for key in defaults:
        if key not in result:
            result[key] = defaults[key]
        elif key != "comment":
            result[key] = max(0, min(5, int(result[key])))

    return result
