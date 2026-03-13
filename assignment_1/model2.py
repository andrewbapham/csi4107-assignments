from sentence_transformers import CrossEncoder
from preprocessing import read_documents_from_file, preprocess_documents, preprocess_text
from indexing import InvertedIndex


# Load the cross-encoder model
ce_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

# Ranking the documents using the cross-encoder model
def cross_encoder_rank(query_text: str, doc_texts: dict[str, str], candidate_doc_ids: list[str]) -> list[tuple[str, float]]:
  pairs = [[query_text, doc_texts[cid]] for cid in candidate_doc_ids]
  # Optimization: Batch the predictions
  scores = ce_model.predict(pairs, batch_size=32, show_progress_bar=True)
  return sorted(zip(candidate_doc_ids, scores), key=lambda x: x[1], reverse=True)

# Filtering the documents that are relevant to the query
def filter_docs(query_text: str, index: InvertedIndex) -> list[str]:
  # Optimization: Only run CE on documents which have at least one term in the query
  candidate_ids = set()
  for term in query_text:
    if term in index.index:
        candidate_ids.update(index.index[term])
  
  return list(candidate_ids)

# Running the cross-encoder model
def run_cross_encoder(query_text: str, documents, index: InvertedIndex) -> list[tuple[str, float]]:
  doc_texts = {
      doc._id: " ".join(doc.title) + " " + " ".join(doc.text)
      for doc in documents
  }
  # Optimization: Query pruning by preprocess_text and filtering the documents that are relevant to the query
  query_text_pruned = preprocess_text(query_text)
  candidate_ids = filter_docs(query_text_pruned, index)
  return cross_encoder_rank(query_text, doc_texts, candidate_ids)

# Testing the cross-encoder model
if __name__ == "__main__":
  raw_documents = read_documents_from_file("scifact/corpus.jsonl")
  documents = preprocess_documents(raw_documents)
  index = InvertedIndex()
  index.build_index(documents)
  query_text = "Can patient-specific iPS cells from Asian individuals with ALS and Parkinson's differentiate into neural cells for disease modeling"
  ranked_docs = run_cross_encoder(query_text, documents, index)
  print(ranked_docs[:10])