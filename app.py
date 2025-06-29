import os
import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

model = SentenceTransformer('all-MiniLM-L6-v2')

hf_token = os.getenv("HF_TOKEN")

generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=hf_token,
    max_length=512,
    do_sample=True,
    temperature=0.7
)

chunks = []
index = None

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join(page.extract_text() for page in reader.pages if page.extract_text())

def split_text(text, max_words=200):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def generate_embeddings(chunks):
    return np.array(model.encode(chunks), dtype=np.float32)

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    return faiss_index

def search_similar_chunks(question, top_k=5):
    q_embedding = model.encode([question]).astype(np.float32)
    distances, indices = index.search(q_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def get_answer_from_llm(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"""Baseado no contexto abaixo, responda de forma objetiva à pergunta.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""

    try:
        response = generator(prompt)
        return response[0]['generated_text']
    except Exception as e:
        return f"Erro ao gerar resposta: {e}"

st.title("Chatbot with PDFs using Mistral model")

uploaded_files = st.file_uploader("📄 Faça upload de PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_texts = [extract_text_from_pdf(f) for f in uploaded_files]
    chunks = []
    for text in all_texts:
        chunks.extend(split_text(text))
    embeddings = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)
    st.success(f"✅ Indexados {len(chunks)} chunks.")

if index:
    question = st.text_input("Digite sua pergunta:")
    if question:
        with st.spinner("🔎 Buscando resposta..."):
            relevant_chunks = search_similar_chunks(question)
            answer = get_answer_from_llm(relevant_chunks, question)
        st.markdown("### 🧠 Resposta:")
        st.write(answer)
