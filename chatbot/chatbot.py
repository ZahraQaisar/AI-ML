from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

# Load FAQs
with open(os.path.join("data", "faqs.json"), "r") as f:
    faqs = json.load(f)

questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]

# Load transformer model and build FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(questions)
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings))

# Terminal chatbot loop
print("🤖 Intern Support Chatbot (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    answer = answers[I[0][0]]
    print("Bot:", answer)
