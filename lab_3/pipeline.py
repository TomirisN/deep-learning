import json
from pathlib import Path
from typing import Callable, List, Optional

from state import AgentState
from tools import invert_abstract, search_openalex, search_wikipedia


def log_step(state: AgentState, action: str, payload: dict, result: str) -> None:
    state.history.append(
        {
            "step_id": state.step_id,
            "action": action,
            "payload": payload,
            "result": result[:300],
        }
    )
    state.step_id += 1


def run_baseline(topic: str, llm_call: Callable[[str], str]) -> str:
    wiki_context = search_wikipedia(topic)
    prompt = f"""
Составьте научный мини-обзор по теме: {topic}.
Используйте следующий контекст:
{wiki_context}
Структура ответа:
1) определение темы
2) ключевые работы
3) 3-5 ключевых идей
4) ограничения
5) перспективы
"""
    return llm_call(prompt)


def _prepare_notes(papers: list, top_n: int) -> List[dict]:
    notes = []
    for paper in papers[:top_n]:
        abstract = invert_abstract(paper.get("abstract_inverted_index"))
        authors = []
        for authorship in paper.get("authorships", [])[:3]:
            author = authorship.get("author", {})
            if author.get("display_name"):
                authors.append(author["display_name"])
        notes.append(
            {
                "title": paper.get("display_name", ""),
                "year": paper.get("publication_year", ""),
                "authors": authors,
                "abstract": abstract[:1200],
            }
        )
    return notes


def run_agent(
    topic: str,
    llm_call: Callable[[str], str],
    max_steps: int = 6,
    per_page: int = 8,
    top_n: int = 5,
    use_evaluator: bool = False,
    evaluate_fn: Optional[Callable] = None,
) -> AgentState:
    state = AgentState(
        topic=topic,
        objective="Сформировать научно-обоснованный мини-обзор темы",
    )

    if state.step_id >= max_steps:
        state.status = "stopped"
        state.stop_reason = "max_steps_reached"
        return state

    wiki_context = search_wikipedia(topic)
    log_step(state, "search_wikipedia", {"topic": topic}, wiki_context)

    if state.step_id >= max_steps:
        state.status = "stopped"
        state.stop_reason = "max_steps_reached"
        return state

    papers = search_openalex(topic, per_page=per_page)
    state.sources = papers
    log_step(
        state,
        "search_openalex",
        {"query": topic, "per_page": per_page},
        f"found={len(papers)}",
    )

    if state.step_id >= max_steps:
        state.status = "stopped"
        state.stop_reason = "max_steps_reached"
        return state

    notes = _prepare_notes(papers, top_n=top_n)
    state.notes = notes
    log_step(state, "extract_notes", {"n_sources": len(notes)}, "notes_prepared")

    if state.step_id >= max_steps:
        state.status = "stopped"
        state.stop_reason = "max_steps_reached"
        return state

    prompt = f"""
Составьте научно-обоснованный мини-обзор по теме: {topic}
Общий контекст:
{wiki_context}
Источники:
{json.dumps(notes, ensure_ascii=False, indent=2)}
Обязательные разделы:
- определение
- ключевые работы
- 3-5 ключевых идей
- ограничения
- перспективы
- использованные источники
Опирайтесь только на предоставленные источники. Не выдумывайте факты.
"""
    state.final_answer = llm_call(prompt)
    log_step(
        state,
        "generate_final_answer",
        {"topic": topic},
        state.final_answer,
    )

    if use_evaluator and evaluate_fn is not None and state.step_id < max_steps:
        eval_result = evaluate_fn(state.final_answer, state.notes, llm_call)
        log_step(
            state,
            "evaluate_draft",
            {"topic": topic},
            json.dumps(eval_result, ensure_ascii=False),
        )

        rubric = (
            eval_result.get("correctness", 0)
            + eval_result.get("groundedness", 0)
            + eval_result.get("completeness", 0)
            + eval_result.get("coverage_of_required_fields", 0)
            + eval_result.get("source_consistency", 0)
        ) / 5.0

        if rubric < 4.0 and state.step_id < max_steps:
            refine_prompt = f"""
Улучшите мини-обзор по теме: {topic}
Текущий черновик:
{state.final_answer}

Замечания evaluator:
{eval_result.get("comment", "")}

Источники:
{json.dumps(notes, ensure_ascii=False, indent=2)}

Сохраните все обязательные разделы и усильте опору на источники.
"""
            state.final_answer = llm_call(refine_prompt)
            log_step(
                state,
                "refine_after_evaluation",
                {"rubric_before": rubric},
                state.final_answer,
            )

    state.status = "finished"
    state.stop_reason = "final_answer_generated"
    return state


def save_trace(state: AgentState, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "topic": state.topic,
                "history": state.history,
                "n_sources": len(state.sources),
                "status": state.status,
                "stop_reason": state.stop_reason,
                "final_answer_preview": state.final_answer[:500],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
