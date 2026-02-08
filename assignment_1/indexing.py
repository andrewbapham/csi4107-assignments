from collections import defaultdict
from typing import Dict
from preprocessing import ScifactDocument

class InvertedIndex:
    def __init__(self):
        # Creating the dictionary in the structure index : Dict[str, Dict[str, int]]
        self.index: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Mapping to document length and frequency
        self.doc_lengths: Dict[str, int] = {}
        self.doc_terms: Dict[str, set[str]] = {}
        self.max_doc_length: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.num_documents = 0

    #Adding a document to the index
    def add_document(self, document: ScifactDocument):
        self.num_documents += 1
        doc_id = document._id
        tokens = document.title + document.text

        # Counting frequency
        term_counts = defaultdict(int)
        max_freq = 0
        for token in tokens:
            term_counts[token] += 1
            max_freq = max(max_freq, term_counts[token])

        self.doc_lengths[doc_id] = len(tokens)
        self.max_doc_length[doc_id] = max_freq
        self.doc_terms[doc_id] = set(tokens)
        # Adding the document terms to the inverted index
        for term, tf in term_counts.items():
            self.index[term][doc_id] = tf
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    # Builds the index
    def build_index(self, documents: list[ScifactDocument]):
        for document in documents:
            self.add_document(document)

    # Returns the posting list for a given term
    def get_postings(self, term: str) -> Dict[str, int]:
        return self.index.get(term, {})

    # Returns the number of unique terms, could be useful in your report?
    def __len__(self):
        return len(self.index)
