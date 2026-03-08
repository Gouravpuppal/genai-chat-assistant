from flask import Flask, request, jsonify, render_template
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
vectors = []


# ----------------------------
# Chunk documents
# ----------------------------
def chunk_text(text, size=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)

    return chunks


# ----------------------------
# Generate embedding
# ----------------------------
def generate_embedding(text):
    return model.encode(text)


# ----------------------------
# Load documents
# ----------------------------
def load_documents():

    with open("docs.json") as f:
        data = json.load(f)

    for doc in data:

        chunks = chunk_text(doc["content"])

        for chunk in chunks:

            embedding = generate_embedding(chunk)

            documents.append(chunk)
            vectors.append(embedding)


# ----------------------------
# Similarity Search (RAG)
# ----------------------------
def search_similar_chunks(query):

    query_embedding = generate_embedding(query)

    similarities = cosine_similarity(
        [query_embedding],
        vectors
    )[0]

    top_indices = similarities.argsort()[-3:][::-1]

    results = []

    for i in top_indices:
        if similarities[i] > 0.3:  # similarity threshold
            results.append(documents[i])

    return results


# ----------------------------
# Generate Response
# ----------------------------
def get_response(chunks, question):

    if not chunks:
        return "Sorry, I couldn't find information related to your question in the knowledge base."

    return chunks[0]


# ----------------------------
# Homepage
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------
# Chat API
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.json
    message = data.get("message")

    chunks = search_similar_chunks(message)

    reply = get_response(chunks, message)

    return jsonify({
        "reply": reply,
        "retrievedChunks": len(chunks)
    })


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":

    print("Loading documents...")

    load_documents()

    app.run(debug=True)