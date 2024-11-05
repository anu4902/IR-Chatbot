import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, create_retrieval_chain
from rank_bm25 import BM25Okapi
from langchain_community.llms import Ollama
import nltk
from nltk.translate.bleu_score import sentence_bleu
import streamlit as st
import time
import numpy as np
from langchain.chains import RetrievalQA
from nltk.corpus import wordnet

# Load data and preprocess
loader = PyPDFLoader("story_long.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Initialize FAISS vector store and BM25 keyword search
db_faiss=FAISS.load_local('/Users/anu/Documents/IR_LAB/Package',OllamaEmbeddings(),allow_dangerous_deserialization=True)
bm25_corpus = [doc.page_content.split() for doc in documents]
bm25 = BM25Okapi(bm25_corpus)

# Load ground truth answers
ground_truth = pd.read_csv("ramayana_qa_dataset.csv")

# Chatbot prompt and LLM setup
prompt = ChatPromptTemplate.from_template("""
You are a chatbot designed to answer questions about the Hindu Mythological epic Ramayana.
Answer the question precisely based on your knowledge only ignore the context. Maintain a neutral tone.
<context>
{context}
</context>
Conversation History:
{conversation_history}
Question: {input}""")

llm = Ollama(model="llama2")

# Create LLMChain and Retrieval Chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db_faiss.as_retriever())

# Hybrid retrieval function
def hybrid_retrieval(query, top_k=5):
    faiss_results = db_faiss.similarity_search(query, k=top_k)
    bm25_query = query.split()
    bm25_scores = bm25.get_scores(bm25_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_results = [documents[i] for i in top_bm25_indices]

    # Combine results, prioritizing FAISS
    seen = set()
    combined_results = []
    for result in faiss_results + bm25_results:
        doc_id = result.metadata.get('id')
        if doc_id not in seen:
            combined_results.append(result)
            seen.add(doc_id)
        if len(combined_results) >= top_k:
            break
    
    return combined_results

# Response generation function with cumulative metric tracking
conversation_history = []

def expand_query(query):
    """Expand the query using synonyms from WordNet."""
    words = query.split()
    expanded_words = set(words)
    new_words=[]  # Use a set to avoid duplicates
    for word in words:
        synonyms = wordnet.synsets(word)
        for synonym in synonyms:
            for lemma in synonym.lemmas():
                new_words.append(lemma.name())  # Add synonyms
    return " ".join(words)

def get_response(query):
    expanded_query = expand_query(query)
    conversation_history.append({"user": query})
    context = "\n".join(f"User: {turn['user']}\nBot: {turn['bot']}" for turn in conversation_history if "bot" in turn)
    
    retrieved_docs = hybrid_retrieval(expanded_query, top_k=5)
    
    # Generate response using the retrieval chain
    response = retrieval_chain({"query": expanded_query, "context": "\n".join(doc.page_content for doc in retrieved_docs), "conversation_history": context})
    
     # Ensure the response has the expected structure
    if 'result' in response:
        conversation_history[-1]["bot"] = response['result']  # Check the correct key for answer
    else:
        conversation_history[-1]["bot"] = "I'm sorry, I couldn't generate a response."

    return conversation_history[-1]["bot"], retrieved_docs

# Streamlit UI
st.title("Ramayana Q&A Chatbot")
st.subheader("Ask any question about the Ramayana!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

user_query = st.text_input("Ask a question about the Ramayana:")

if st.button("Send"):
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.spinner("Typing..."):
            bot_response, retrieved_docs = get_response(user_query)
            time.sleep(1)
            st.session_state.messages.append({"role": "bot", "content": bot_response})


# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")
