from preprocessing import read_documents_from_file, preprocess_documents
from indexing import InvertedIndex

# Initial Load
documents = read_documents_from_file("scifact/corpus.jsonl")

# Testing Part 1
documents = preprocess_documents(documents)

# Testing Part 2
index = InvertedIndex()
index.build_index(documents)
print("Number of documents:", index.num_documents)
print("Vocabulary size:", len(index))

# Check how many times the term appears in documents
term = "climat"  
postings = index.get_postings(term)
print(f"Postings for '{term}':")
print(f"Number of documents containing term: {len(postings)}")

# Comparing the index to manual counting to make sure it works
document = documents[0]
term = document.text[0]
manual = document.title.count(term) + document.text.count(term)
index_num = index.index[term][document._id]

# Printing so we can see if they match
print("Manually Found:", manual)
print("Index Found:", index_num)
