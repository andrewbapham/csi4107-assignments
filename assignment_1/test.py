
from preprocessing import read_documents_from_file, preprocess_documents
from indexing import InvertedIndex
from utils import build_query_tf_idf_vector, build_doc_tf_idf_vector, get_magnitude, get_cosine_similarity
from collections import defaultdict

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

# Testing Part 3
print(len(index.doc_terms))
query = "the method of reducing the risk of climate change"
doc_id_1 = '4810810'
doc_id_2 = '36178047'
doc_id_3 = '13882658'

query_vec = build_query_tf_idf_vector(query, index)
query_mag = get_magnitude(query_vec)

doc_vec_1 = build_doc_tf_idf_vector(doc_id_1, index)
doc_mag_1 = get_magnitude(doc_vec_1)
doc_vec_2 = build_doc_tf_idf_vector(doc_id_2, index)
doc_mag_2 = get_magnitude(doc_vec_2)
doc_vec_3 = build_doc_tf_idf_vector(doc_id_3, index)
doc_mag_3 = get_magnitude(doc_vec_3)

# Document 3 (13882658) should have a higher cosine similarity than document 1 (4810810) and 2 (36178047)
print("Cosine Similarity between query and document 1 (4810810):", get_cosine_similarity(query_vec, doc_vec_1, query_mag, doc_mag_1))
print("Cosine Similarity between query and document 2 (36178047):", get_cosine_similarity(query_vec, doc_vec_2, query_mag, doc_mag_2))
print("Cosine Similarity between query and document 3 (13882658):", get_cosine_similarity(query_vec, doc_vec_3, query_mag, doc_mag_3))




#Reusing logic from above
def rank_documents(query_text, index, top_k=100):
    query_vec = build_query_tf_idf_vector(query_text, index)
    query_mag = get_magnitude(query_vec)

    scores = []
    for doc_id in index.doc_terms:
        doc_vec = build_doc_tf_idf_vector(doc_id, index)
        doc_mag = get_magnitude(doc_vec)

        if doc_mag == 0:
            continue

        score = get_cosine_similarity(query_vec, doc_vec, query_mag, doc_mag)

        if score > 0:
            scores.append((doc_id, score))
#Sorting the list of doc_id , score pairs by the score (second element)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


import json



queries = []
#Grab the queries from queries.jsonl
with open("scifact/queries.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        qid = int(q["_id"])
        if qid % 2 == 1:
            queries.append(q)

print("Test queries loaded:", len(queries))



#Writing to the results file with the required format
with open("Results", "w", encoding="utf-8") as out:
    for q in queries:
        qid = q["_id"]
        query_text = q["text"]

        ranked_docs = rank_documents(query_text, index, top_k=100)

        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} run1\n")




# Only used for report
#with open("Results_title_only", "w", encoding="utf-8") as out:
 #   for q in queries:
  #    ranked_docs = rank_documents(query_text, index, top_k=100)
   #     for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
    #        out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} run_title\n")

    #vocabulary = list(index.index.keys())
    #print("Vocabulary size:", len(vocabulary))

    #rint("Sample 100 tokens from vocabulary:", vocabulary[:100])

