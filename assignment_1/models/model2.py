from sentence_transformers import CrossEncoder
import tqdm

from preprocessing import read_documents_from_file, preprocess_documents
from indexing import InvertedIndex
from utils import (
    build_query_tf_idf_vector,
    build_doc_tf_idf_vector,
    get_magnitude,
    get_cosine_similarity,
)


# Load the cross-encoder model once.
ce_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")


def rank_documents(query_text, index, doc_vectors, doc_magnitudes, top_k=100):
    """First-stage retrieval with TF-IDF cosine similarity."""
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

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def cross_encoder_rank(
    query_text: str, doc_texts: dict[str, str], candidate_doc_ids: list[str]
) -> list[tuple[str, float]]:
    """Second-stage reranking with CrossEncoder over retrieved candidates."""
    if not candidate_doc_ids:
        return []
    pairs = [[query_text, doc_texts[cid]] for cid in candidate_doc_ids]
    scores = ce_model.predict(pairs, batch_size=32)
    return sorted(zip(candidate_doc_ids, scores), key=lambda x: x[1], reverse=True)


def build_doc_texts(raw_documents) -> dict[str, str]:
    """Use original (unprocessed) document text for the cross-encoder."""
    return {doc._id: doc.title + " " + doc.text for doc in raw_documents}


def build_model2_resources(corpus_path: str = "scifact/corpus.jsonl"):
    raw_documents = read_documents_from_file(corpus_path)
    documents = preprocess_documents(read_documents_from_file(corpus_path))
    index = InvertedIndex()
    index.build_index(documents)

    doc_vectors = {}
    doc_magnitudes = {}
    for doc_id in index.doc_terms:
        vec = build_doc_tf_idf_vector(doc_id, index)
        doc_vectors[doc_id] = vec
        doc_magnitudes[doc_id] = get_magnitude(vec)

    doc_texts = build_doc_texts(raw_documents)
    return index, doc_vectors, doc_magnitudes, doc_texts


def run_model2(
    queries,
    output_path: str = "Results_Model2",
    top_k: int = 100,
    corpus_path: str = "scifact/corpus.jsonl",
):
    index, doc_vectors, doc_magnitudes, doc_texts = build_model2_resources(corpus_path)

    run = {}
    with open(output_path, "w", encoding="utf-8") as out:
        for q in tqdm.tqdm(queries, desc="Running model 2", unit="queries"):
            qid = q["_id"] if isinstance(q, dict) else q[0]
            query_text = q["text"] if isinstance(q, dict) else q[1]

            top_docs = rank_documents(
                query_text, index, doc_vectors, doc_magnitudes, top_k=top_k
            )
            candidate_ids = [doc_id for doc_id, _score in top_docs]
            ranked_docs = cross_encoder_rank(query_text, doc_texts, candidate_ids)

            run[qid] = [doc_id for doc_id, _score in ranked_docs]
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} CrossEncoder\n")

    return run


if __name__ == "__main__":
    query_text = (
        "Can patient-specific iPS cells from Asian individuals with ALS and "
        "Parkinson's differentiate into neural cells for disease modeling"
    )
    index, doc_vectors, doc_magnitudes, doc_texts = build_model2_resources(
        "scifact/corpus.jsonl"
    )
    top_docs = rank_documents(query_text, index, doc_vectors, doc_magnitudes, top_k=100)
    candidate_ids = [doc_id for doc_id, _score in top_docs]
    ranked_docs = cross_encoder_rank(query_text, doc_texts, candidate_ids)
    print(ranked_docs[:10])
