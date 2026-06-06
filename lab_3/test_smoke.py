"""Проверка API и импортов без вызова LLM."""

from pipeline import run_baseline, run_agent, save_trace
from state import AgentState
from tools import invert_abstract, search_openalex, search_wikipedia


def mock_llm(prompt: str) -> str:
    return (
        "1) Определение: тестовый мини-обзор.\n"
        "2) Ключевые работы: см. источники.\n"
        "3) Идеи: RAG, agents, evaluation.\n"
        "4) Ограничения: мало данных.\n"
        "5) Перспективы: масштабирование.\n"
        '{"correctness": 4, "groundedness": 4, "completeness": 4, '
        '"coverage_of_required_fields": 4, "source_consistency": 4, "comment": "ok"}'
    )


def main() -> None:
    topic = "LLM evaluation and process-aware metrics"
    wiki = search_wikipedia(topic)
    papers = search_openalex(topic, per_page=3)
    assert len(papers) > 0, "OpenAlex вернул пустой результат"
    print(f"Wikipedia: {len(wiki)} символов")
    print(f"OpenAlex: {len(papers)} статей")

    baseline = run_baseline(topic, mock_llm)
    assert "мини-обзор" in baseline.lower() or "обзор" in baseline.lower()
    print("Baseline OK")

    state = run_agent(topic, mock_llm, max_steps=6, per_page=5, top_n=3)
    assert state.status == "finished"
    assert len(state.history) >= 4
    save_trace(state, "outputs/traces/test_trace.json")
    print(f"Agent OK, steps={len(state.history)}")

    from evaluator import evaluate_answer

    ev = evaluate_answer(state.final_answer, state.notes, mock_llm)
    assert ev["correctness"] >= 0
    print(f"Evaluator OK, rubric~={(ev['correctness']+ev['groundedness'])/2}")
    print("\nВсе проверки пройдены.")


if __name__ == "__main__":
    main()
