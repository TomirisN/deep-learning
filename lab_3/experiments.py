import json
import time
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from config import OUTPUT_DIR, TOPICS
from evaluator import evaluate_answer
from pipeline import run_agent, run_baseline, save_trace
from state import AgentState


def run_experiment(
    topics: List[str],
    mode_name: str,
    runner: Callable,
    llm_call: Callable[[str], str],
    save_traces: bool = True,
) -> List[dict]:
    records = []
    trace_dir = Path(OUTPUT_DIR) / "traces" / mode_name
    trace_dir.mkdir(parents=True, exist_ok=True)

    for topic in topics:
        print(f"  [{mode_name}] {topic}...", flush=True)
        start = time.time()
        result = runner(topic, llm_call)
        latency = time.time() - start

        if isinstance(result, AgentState):
            answer = result.final_answer
            notes = result.notes
            n_steps = len(result.history)
            if save_traces:
                safe_name = topic.replace(" ", "_").replace("/", "-")[:80]
                save_trace(result, str(trace_dir / f"{safe_name}.json"))
        else:
            answer = result
            notes = []
            n_steps = 1

        eval_json = evaluate_answer(answer, notes, llm_call)
        records.append(
            {
                "topic": topic,
                "mode": mode_name,
                "correctness": eval_json["correctness"],
                "groundedness": eval_json["groundedness"],
                "completeness": eval_json["completeness"],
                "coverage": eval_json["coverage_of_required_fields"],
                "source_consistency": eval_json["source_consistency"],
                "rubric": sum(
                    [
                        eval_json["correctness"],
                        eval_json["groundedness"],
                        eval_json["completeness"],
                        eval_json["coverage_of_required_fields"],
                        eval_json["source_consistency"],
                    ]
                )
                / 5.0,
                "n_steps": n_steps,
                "latency": latency,
                "eval_comment": eval_json.get("comment", ""),
            }
        )

    return records


def run_all_experiments(
    llm_call: Callable[[str], str],
    topics: Optional[List[str]] = None,
    quick: bool = False,
) -> pd.DataFrame:
    topics = topics or TOPICS
    if quick:
        topics = topics[:2]

    all_records: List[dict] = []

    print("\n=== Конфигурация 1: Baseline ===")
    all_records += run_experiment(topics, "baseline", run_baseline, llm_call)

    print("\n=== Конфигурация 2: Agent (без evaluator в цикле) ===")
    agent_runner = partial(run_agent, use_evaluator=False)
    all_records += run_experiment(topics, "agent", agent_runner, llm_call)

    print("\n=== Конфигурация 3: Agent + Evaluator ===")
    agent_eval_runner = partial(
        run_agent,
        use_evaluator=True,
        evaluate_fn=evaluate_answer,
    )
    all_records += run_experiment(topics, "agent+evaluator", agent_eval_runner, llm_call)

    if not quick:
        print("\n=== Конфигурация 4: Agent — число источников (top-3, top-5, top-8) ===")
        for top_n in (3, 5, 8):
            runner = partial(run_agent, top_n=top_n, per_page=max(top_n, 5), use_evaluator=False)
            all_records += run_experiment(topics, f"agent_top{top_n}", runner, llm_call)

        print("\n=== Конфигурация 5: Agent — max_steps (4, 6, 8) ===")
        for max_steps in (4, 6, 8):
            runner = partial(run_agent, max_steps=max_steps, use_evaluator=False)
            all_records += run_experiment(topics, f"agent_steps{max_steps}", runner, llm_call)

    df = pd.DataFrame(all_records)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")

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
    summary.to_csv(out_dir / "summary.csv", encoding="utf-8-sig")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.round(3).to_dict(), f, ensure_ascii=False, indent=2)

    return df


def plot_results(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    out_dir = Path(OUTPUT_DIR)
    main_modes = ["baseline", "agent", "agent+evaluator"]
    plot_df = df[df["mode"].isin(main_modes)].groupby("mode")[
        ["correctness", "groundedness", "completeness", "rubric"]
    ].mean()

    if plot_df.empty:
        plot_df = df.groupby("mode")[
            ["correctness", "groundedness", "completeness", "rubric"]
        ].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Сравнение режимов: качество ответов")
    ax.set_ylabel("Средний балл (0–5)")
    ax.set_xlabel("Режим")
    ax.legend(loc="lower right")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "quality_comparison.png", dpi=150)
    plt.close()

    perf_df = df[df["mode"].isin(main_modes)].groupby("mode")[
        ["n_steps", "latency"]
    ].mean()
    if perf_df.empty:
        perf_df = df.groupby("mode")[["n_steps", "latency"]].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    perf_df.plot(kind="bar", ax=ax)
    ax.set_title("Сравнение режимов: стоимость выполнения")
    ax.set_ylabel("Шаги / секунды")
    ax.set_xlabel("Режим")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "performance_comparison.png", dpi=150)
    plt.close()

    print(f"\nГрафики сохранены в {out_dir}/")
