import math
import numpy as np
from indexing import InvertedIndex
from collections import defaultdict
from preprocessing import preprocess_text

# Functions for TF-IDF
def get_tf(term_freq: int, max_freq: int) -> float:
    return term_freq / max_freq

import math

def get_idf(doc_freq, total_docs):
    if doc_freq == 0:
        return 0
    return math.log2(total_docs / doc_freq)


def get_tf_idf(term_freq: int, max_freq: int, doc_freq: int, total_docs: int) -> float:
    tf = get_tf(term_freq, max_freq)
    idf = get_idf(doc_freq, total_docs)
    return tf * idf

# Functions for Cosine Similarity
def get_dot_product(query_vec: dict[str, int], doc_vec: dict[str, int]) -> float:
    return sum(query_vec.get(term, 0) * doc_vec.get(term, 0) for term in query_vec)

def get_magnitude(vec: dict[str, int]) -> float:
    return math.sqrt(sum(w**2 for w in vec.values()))

def get_cosine_similarity(query_vec: dict[str, int], doc_vec: dict[str, int], query_mag: float, doc_mag: float) -> float:
    dot_product = get_dot_product(query_vec, doc_vec)
    if query_mag * doc_mag == 0: return 0
    return dot_product / (query_mag * doc_mag)

# Functions for building the TF-IDF vectors
def build_query_tf_idf_vector(query: str, index: InvertedIndex) -> dict[str, int]:
    tokens = preprocess_text(query)
    query_vec = {}
    query_term_counts = defaultdict(int)
    query_max_freq = 0
    # Counting the frequency of the query terms
    for token in tokens:
        query_term_counts[token] += 1
        query_max_freq = max(query_max_freq, query_term_counts[token])

    # Building the query TF-IDF vector
    for term, tf in query_term_counts.items():
        doc_freq = index.doc_freqs.get(term, 0)
        total_docs = index.num_documents
        query_vec[term] = get_tf(tf, query_max_freq) * get_idf(doc_freq, total_docs)
    return query_vec


def build_doc_tf_idf_vector(doc_id: str, index: InvertedIndex) -> dict[str, int]:
    doc_vec = {}
    doc_terms = index.doc_terms[doc_id]
    
    # Building the document TF-IDF vector
    for term in doc_terms:
        term_freq = index.index[term][doc_id]
        max_freq = index.max_doc_length[doc_id]
        doc_freq = index.doc_freqs[term]
        total_docs = index.num_documents
        doc_vec[term] = get_tf(term_freq, max_freq) * get_idf(doc_freq, total_docs)
    
    return doc_vec
