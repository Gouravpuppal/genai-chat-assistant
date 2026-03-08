# Production-Grade GenAI Assistant with RAG

## Overview

This project implements a GenAI-powered chat assistant using Retrieval-Augmented Generation (RAG). The assistant retrieves relevant information from a document knowledge base and generates responses grounded in that context.

## Tech Stack

* Backend: Python (Flask)
* Frontend: HTML, CSS, JavaScript
* Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
* Vector Similarity: Cosine Similarity
* Storage: In-memory vector store

## Project Architecture

User Query
↓
Convert Query to Embedding
↓
Similarity Search (Vector DB)
↓
Retrieve Top Relevant Chunks
↓
Inject Context into Prompt
↓
Generate Grounded Response

## RAG Workflow

1. Documents are stored in docs.json
2. Documents are split into smaller chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in memory
5. User question is converted to embedding
6. Cosine similarity retrieves the most relevant chunks
7. Retrieved context is used to generate the final answer

## Embedding Strategy

We use the SentenceTransformer model **all-MiniLM-L6-v2** to convert text into numerical embeddings.

## Similarity Search

Cosine similarity is used to compare user query embeddings with document embeddings.

Top 3 relevant chunks are retrieved.

## Prompt Design

Responses are grounded using retrieved context to avoid hallucinated answers.

If similarity is below threshold, the assistant returns a fallback message.

## Setup Instructions

### Install dependencies

pip install flask sentence-transformers scikit-learn

### Run the project

python app.py

### Open in browser

http://localhost:5000

## Project Structure

genai-chat-assistant
│
├── app.py
├── docs.json
├── requirements.txt
│
├── templates
│   └── index.html
│
└── static
├── script.js
└── styles.css

## Example Query

User: How can I reset my password?
Assistant: Users can reset their password from Settings > Security.

## Features

* Document embedding
* Similarity search
* Retrieval-Augmented Generation
* Chat interface
* Flask API endpoint
