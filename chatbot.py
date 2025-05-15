import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Step 1: Load FAQ and Support Ticket Data
with open('faq_support_data.json', 'r') as file:
    data = json.load(file)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

# Step 2: Vectorize all questions using TF-IDF
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Step 3: Load Hugging Face QA pipeline (optional for large answers)
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Step 4: Chat Loop
print("\n🤖 Internship Support Chatbot is ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Step 5: Find closest question using cosine similarity
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarities.argmax()

    best_question = questions[best_match_index]
    best_answer = answers[best_match_index]
    confidence = similarities[0][best_match_index]

    # Show result
    print(f"\n🤖 Answer (matched question: '{best_question}')")
    print(f"💬 {best_answer} (Confidence: {confidence:.2f})\n")
