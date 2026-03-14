from pathlib import Path
import json

import tqdm

from models.model1 import run_model1
from models.model2 import run_model2


def load_qrels(qrels_path: Path) -> dict[str, set[str]]:
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if idx == 0 and "query-id" in line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            qid, doc_id = parts[0], parts[1]
            qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def load_queries(queries_path: Path, qrels: dict[str, set[str]]) -> list[dict]:
    query_ids = set(qrels.keys())
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if q["_id"] in query_ids:
                queries.append(q)
    return queries


def evaluate_run(
    run: dict[str, list[str]], qrels: dict[str, set[str]], k: int = 10
) -> tuple[float, float]:
    avg_precision_scores = []
    p10_scores = []

    for qid, relevant in qrels.items():
        ranked_docs = run.get(qid, [])

        top_k = ranked_docs[:k]
        p10 = sum(1 for doc_id in top_k if doc_id in relevant) / k
        p10_scores.append(p10)

        hits = 0
        sum_precision = 0.0
        for i, doc_id in enumerate(ranked_docs):
            if doc_id in relevant:
                hits += 1
                sum_precision += hits / (i + 1)
        ap = sum_precision / len(relevant) if relevant else 0.0
        avg_precision_scores.append(ap)

    map_score = (
        sum(avg_precision_scores) / len(avg_precision_scores)
        if avg_precision_scores
        else 0.0
    )
    p10_score = sum(p10_scores) / len(p10_scores) if p10_scores else 0.0
    return map_score, p10_score


def print_eval_results(
    map_score: float, p10_score: float, model_name: str, model_description: str
):
    print("--------------------------------")
    print(f"Eval summary for {model_name} ({model_description})")
    print(f"MAP:  {map_score:.4f}")
    print(f"P@10: {p10_score:.4f}")
    print("--------------------------------")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    corpus_path = base_dir / "scifact" / "corpus.jsonl"
    queries_path = base_dir / "scifact" / "queries.jsonl"
    qrels_path = base_dir / "scifact" / "qrels" / "test.tsv"

    qrels = load_qrels(qrels_path)
    queries = load_queries(queries_path, qrels)
    print(f"Loaded {len(queries)} queries with qrels")

    run1_path = base_dir / "Results_Model1"
    run1 = run_model1(
        queries,
        output_path=str(run1_path),
        top_k=100,
        corpus_path=str(corpus_path),
    )
    map1, p10_1 = evaluate_run(run1, qrels)
    print_eval_results(map1, p10_1, "Model1", "SBERT rerank")

    run2_path = base_dir / "Results_Model2"
    run2 = run_model2(
        queries,
        output_path=str(run2_path),
        top_k=100,
        corpus_path=str(corpus_path),
    )
    map2, p10_2 = evaluate_run(run2, qrels)
    print_eval_results(map2, p10_2, "Model2", "Cross-Encoder")


if __name__ == "__main__":
    main()
