import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from dataclasses import dataclass, asdict
import json
import tqdm
import nltk

nltk.download("punkt_tab")


@dataclass
class ScifactDocument:
    _id: str
    title: str
    text: str
    metadata: dict


@dataclass
class ScifactQuery:
    _id: str
    text: str
    metadata: dict


def read_stopwords(file_path: str) -> set[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file.readlines())


stop_words = read_stopwords("stopwords.txt")
stemmer = PorterStemmer()
STOPWORDS_REMOVED = 0


def reset_stopword_counter() -> None:
    global STOPWORDS_REMOVED
    STOPWORDS_REMOVED = 0


def get_stopword_counter() -> int:
    return STOPWORDS_REMOVED


def strip_markup(text: str) -> str:
    # Remove simple XML/HTML-style tags and collapse whitespace.
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return word_tokenize(strip_markup(text).lower())


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stemmer.stem(token) for token in tokens]


def is_valid_token(token: str) -> bool:
    if not token:
        return False
    if token in stop_words:
        return False
    # maybe later filter to just alphabetic tokens?
    return True


def remove_extras(tokens: list[str]) -> list[str]:
    global STOPWORDS_REMOVED
    STOPWORDS_REMOVED += sum(1 for token in tokens if token in stop_words)
    return [token for token in tokens if is_valid_token(token)]


def preprocess_text(text: str, use_stemming: bool = True) -> list[str]:
    tokens = tokenize(text)

    tokens = remove_extras(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    return tokens


def preprocess_document(document: ScifactDocument) -> ScifactDocument:
    document.title = preprocess_text(document.title)
    document.text = preprocess_text(document.text)
    return document


def preprocess_documents(documents: list[ScifactDocument]) -> list[ScifactDocument]:
    return [
        preprocess_document(document)
        for document in tqdm.tqdm(
            documents, desc="Preprocessing documents", unit="documents"
        )
    ]


def preprocess_query(query: ScifactQuery) -> ScifactQuery:
    query.text = preprocess_text(query.text)
    return query


def read_documents_from_file(document_file: str) -> list[ScifactDocument]:
    with open(document_file, "r", encoding="utf-8") as file:
        return [
            ScifactDocument(**json.loads(line.strip()))
            for line in file.readlines()
            if line.strip()
        ]


def save_documents_to_file(documents: list[ScifactDocument], file_path: str):
    with open(file_path, "w", encoding="utf-8") as file:
        for document in tqdm.tqdm(documents, desc="Saving documents", unit="documents"):
            document.text = " ".join(document.text)
            document.title = " ".join(document.title)
            file.write(json.dumps(asdict(document)) + "\n")
            file.flush()


if __name__ == "__main__":
    reset_stopword_counter()
    documents = read_documents_from_file("scifact/corpus.jsonl")
    preprocessed_documents = preprocess_documents(documents)
    # save_documents_to_file(preprocessed_documents, "scifact/corpus_preprocessed.jsonl")
    print(f"Total stopwords removed: {get_stopword_counter()}")
