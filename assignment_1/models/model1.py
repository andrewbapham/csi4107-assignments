import json

import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import read_documents_from_file, preprocess_documents
from indexing import InvertedIndex
from utils import (
    build_query_tf_idf_vector,
    build_doc_tf_idf_vector,
    get_magnitude,
    get_cosine_similarity,
)


# Reusing lots of code from a1 here
def rank_documents(query_text, index, doc_vectors, doc_magnitudes, top_k=100):
    scores = []
    query_vec = build_query_tf_idf_vector(query_text, index)
    query_mag = get_magnitude(query_vec)
    for doc_id in doc_vectors:
        doc_vec = doc_vectors[doc_id]
        doc_mag = doc_magnitudes[doc_id]

        if doc_mag == 0:
            continue

        score = get_cosine_similarity(query_vec, doc_vec, query_mag, doc_mag)

        if score > 0:
            scores.append((doc_id, score))
    # Sorting the list of doc_id , score pairs by the score (second element)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_model1_resources(corpus_path: str = "scifact/corpus.jsonl"):
    raw_documents = read_documents_from_file(corpus_path)
    documents = preprocess_documents(read_documents_from_file(corpus_path))
    index = InvertedIndex()
    index.build_index(documents)

    # Precomputing TF-IDF vectors for all documents to help with optimization
    doc_vectors = {}
    doc_magnitudes = {}
    for doc_id in index.doc_terms:
        vec = build_doc_tf_idf_vector(doc_id, index)
        doc_vectors[doc_id] = vec
        doc_magnitudes[doc_id] = get_magnitude(vec)

    doc_texts = {}
    for doc in raw_documents:
        doc_texts[doc._id] = doc.title + " " + doc.text

    print("Loading SBERT ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding documents with all-MiniLM-L6-v2")
    doc_ids = list(doc_texts.keys())
    doc_embeddings = model.encode(
        [doc_texts[d] for d in doc_ids], show_progress_bar=True
    )
    id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    return index, doc_vectors, doc_magnitudes, model, doc_embeddings, id_to_index


# Rerank
def sbert_rerank(query_text, candidate_doc_ids, model, doc_embeddings, id_to_index):
    query_vec = model.encode([query_text])
    candidate_ids = [doc_id for doc_id in candidate_doc_ids if doc_id in id_to_index]
    if not candidate_ids:
        return []
    ind = [id_to_index[doc_id] for doc_id in candidate_ids]
    candidate_emb = doc_embeddings[ind]
    scores = cosine_similarity(query_vec, candidate_emb)[0]
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked


def run_model1(
    queries,
    output_path: str = "Results_SBERT",
    top_k: int = 100,
    corpus_path: str = "scifact/corpus.jsonl",
):
    index, doc_vectors, doc_magnitudes, model, doc_embeddings, id_to_index = (
        build_model1_resources(corpus_path)
    )

    run = {}
    with open(output_path, "w", encoding="utf-8") as out:
        for q in tqdm.tqdm(queries, desc="Running model 1", unit="queries"):
            qid = q["_id"] if isinstance(q, dict) else q[0]
            query_text = q["text"] if isinstance(q, dict) else q[1]
            top_100 = rank_documents(
                query_text, index, doc_vectors, doc_magnitudes, top_k=top_k
            )
            top_100_ids = [doc_id for doc_id, score in top_100]
            reranked = sbert_rerank(
                query_text, top_100_ids, model, doc_embeddings, id_to_index
            )

            run[qid] = [doc_id for doc_id, _score in reranked]
            for rank, (doc_id, score) in enumerate(reranked, start=1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} SBERT\n")

    return run


# Evaluation just so I could see the MAP score myself
def evaluate(results_file, qrels_file):
    qrels = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if idx == 0 and "query-id" in line:
                continue
            parts = line.split()
            qid, doc_id = parts[0], parts[1]
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(doc_id)

    results = {}
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            qid, doc_id, rank = parts[0], parts[2], int(parts[3])
            if qid not in results:
                results[qid] = []
            results[qid].append((rank, doc_id))

    for qid in results:
        results[qid].sort(key=lambda x: x[0])

    ap_scores = []
    p10_scores = []

    for qid in qrels:
        if qid not in results:
            ap_scores.append(0)
            p10_scores.append(0)
            continue

        relevant = qrels[qid]
        ranked_docs = [doc_id for rank, doc_id in results[qid]]
        # p10
        top10 = ranked_docs[:10]
        p10 = sum(1 for doc in top10 if doc in relevant) / 10
        p10_scores.append(p10)
        # map
        hits = 0
        sum_precision = 0
        for i, doc_id in enumerate(ranked_docs):
            if doc_id in relevant:
                hits += 1
                sum_precision += hits / (i + 1)
        ap = sum_precision / len(relevant) if relevant else 0
        ap_scores.append(ap)

    map_score = sum(ap_scores) / len(ap_scores)
    p10_score = sum(p10_scores) / len(p10_scores)

    print(f"MAP:  {map_score:.4f}")
    print(f"P@10: {p10_score:.4f}")


def main():
    # retrieve the queries
    queries = []
    with open("scifact/queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            qid = int(q["_id"])
            if qid % 2 == 1:
                queries.append(q)

    print(f"Loaded {len(queries)} queries")
    run_model1(queries, output_path="Results_SBERT")
    evaluate("Results_SBERT", "scifact/qrels/test.tsv")
    evaluate("Results", "scifact/qrels/test.tsv")


if __name__ == "__main__":
    main()
