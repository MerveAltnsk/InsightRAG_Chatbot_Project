import streamlit as st
import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import time
from datetime import datetime
import uuid
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import re

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Chatbot: ML/AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .bot-message {
        background-color: #e8f4fd;
        border-left-color: #764ba2;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# RAG System Functions (from notebook)
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only keep substantial chunks
            chunks.append(chunk)
    
    return chunks

def load_and_process_dataset():
    """Load and process The Pile dataset"""
    print("📚 Loading The Pile dataset...")
    
    try:
        # Load a specific subset that contains ML/AI content
        dataset = load_dataset("EleutherAI/the_pile", split="train", streaming=True)
        
        # Take first 1000 samples for demonstration
        texts = []
        ml_keywords = ['machine learning', 'deep learning', 'neural network', 'artificial intelligence', 
                       'algorithm', 'model', 'training', 'data', 'feature', 'classification', 
                       'regression', 'clustering', 'optimization', 'gradient', 'tensor']
        
        print("🔍 Filtering ML/AI related content...")
        count = 0
        for sample in tqdm(dataset, desc="Processing samples"):
            if count >= 1000:  # Limit to 1000 samples for demo
                break
                
            text = sample['text']
            # Check if text contains ML/AI keywords
            if any(keyword in text.lower() for keyword in ml_keywords):
                # Clean and preprocess text
                text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
                text = text.strip()
                
                # Only keep texts that are reasonable length (not too short or too long)
                if 100 <= len(text) <= 2000:
                    texts.append(text)
                    count += 1
        
        print(f"✅ Loaded {len(texts)} ML/AI related text samples")
        return texts
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("🔄 Using fallback sample data...")
        
        # Fallback sample data if The Pile is not accessible
        texts = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. Deep learning uses neural networks with multiple layers to process complex patterns in data.",
            "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using a connectionist approach.",
            "Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common algorithms include linear regression, decision trees, and support vector machines.",
            "Unsupervised learning finds hidden patterns in data without labeled examples. Clustering algorithms like K-means group similar data points together.",
            "Natural language processing combines computational linguistics with machine learning to help computers understand human language. It includes tasks like text classification and sentiment analysis.",
            "Computer vision enables machines to interpret and understand visual information from the world. It uses deep learning models like convolutional neural networks.",
            "Reinforcement learning is a type of machine learning where agents learn to make decisions by interacting with an environment and receiving rewards or penalties.",
            "Feature engineering is the process of selecting and transforming raw data into features that can be used by machine learning algorithms. Good features can significantly improve model performance.",
            "Cross-validation is a technique used to assess how well a machine learning model generalizes to new data. It involves splitting data into training and validation sets multiple times.",
            "Overfitting occurs when a model learns the training data too well and performs poorly on new data. Regularization techniques help prevent overfitting."
        ]
        print(f"✅ Using {len(texts)} sample texts")
        return texts

def initialize_rag_system(api_key):
    """Initialize the RAG system with all components"""
    try:
        # Set API key
        os.environ['GOOGLE_API_KEY'] = api_key
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Chroma
        chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        collection_name = "ml_ai_knowledge"
        try:
            collection = chroma_client.get_collection(collection_name)
            print(f"✅ Found existing collection: {collection_name}")
        except:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "ML/AI knowledge base from The Pile dataset"}
            )
            print(f"✅ Created new collection: {collection_name}")
        
        # Check if collection already has data
        existing_count = collection.count()
        print(f"📊 Current documents in collection: {existing_count}")
        
        if existing_count == 0:
            print("🔄 Adding new documents to collection...")
            
            # Load and process dataset
            texts = load_and_process_dataset()
            
            all_chunks = []
            chunk_ids = []
            chunk_metadatas = []
            
            for i, text in enumerate(tqdm(texts, desc="Processing texts")):
                chunks = chunk_text(text)
                
                for j, chunk in enumerate(chunks):
                    chunk_id = f"doc_{i}_chunk_{j}"
                    metadata = {
                        "source": f"the_pile_doc_{i}",
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "text_length": len(chunk)
                    }
                    
                    all_chunks.append(chunk)
                    chunk_ids.append(chunk_id)
                    chunk_metadatas.append(metadata)
            
            print(f"📊 Created {len(all_chunks)} text chunks")
            
            # Add documents to Chroma in batches to avoid memory issues
            batch_size = 100
            for i in tqdm(range(0, len(all_chunks), batch_size), desc="Adding to Chroma"):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                batch_metadatas = chunk_metadatas[i:i + batch_size]
                
                collection.add(
                    documents=batch_chunks,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
            
            print("✅ All documents added to Chroma!")
        else:
            print("✅ Collection already contains data, skipping addition")
        
        # Initialize Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.7,
            max_output_tokens=1024,
            convert_system_message_to_human=True
        )
        
        return {
            'embedding_model': embedding_model,
            'chroma_client': chroma_client,
            'collection': collection,
            'llm': llm
        }
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

def retrieve_relevant_docs(query, collection, n_results=5):
    """Retrieve relevant documents from Chroma"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Extract documents and metadata
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        return documents, metadatas, distances
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return [], [], []

def create_context(documents):
    """Create context string from retrieved documents"""
    context = "\n\n".join(documents)
    return context

def generate_answer(query, context, llm):
    """Generate answer using Gemini with retrieved context"""
    system_prompt = """You are an AI assistant specialized in machine learning, deep learning, and artificial intelligence. 
    Use the provided context to answer questions accurately and comprehensively. If the context doesn't contain enough 
    information, you can supplement with your general knowledge, but always prioritize the provided context.
    
    Provide clear, well-structured answers with examples when appropriate."""
    
    user_prompt = f"""Context:
    {context}
    
    Question: {query}
    
    Please provide a comprehensive answer based on the context above."""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating answer: {e}"

def rag_pipeline(query, rag_system, n_results=5):
    """Complete RAG pipeline"""
    try:
        collection = rag_system['collection']
        llm = rag_system['llm']
        
        # Retrieve relevant documents
        documents, metadatas, distances = retrieve_relevant_docs(query, collection, n_results)
        
        if not documents:
            return "I couldn't find relevant information for your query. Please try asking about machine learning, deep learning, or AI topics."
        
        # Create context
        context = create_context(documents)
        
        # Generate answer
        answer = generate_answer(query, context, llm)
        return answer, documents, distances
        
    except Exception as e:
        return f"Error generating response: {e}", [], []

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Chatbot: ML/AI Assistant</h1>
    <p>Powered by Google Gemini 2.5 Flash + LangChain + Chroma</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        help="Get your API key from Google AI Studio"
    )
    
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
    
    # Initialize button
    if st.button("🚀 Initialize RAG System", disabled=not api_key):
        with st.spinner("Initializing RAG system..."):
            try:
                rag_system = initialize_rag_system(api_key)
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True
                    st.success("✅ RAG system initialized successfully!")
                else:
                    st.error("❌ Failed to initialize system")
            except Exception as e:
                st.error(f"❌ Error initializing system: {e}")
    
    # System status
    st.markdown("## 📊 System Status")
    if st.session_state.initialized:
        st.success("🟢 System Ready")
        try:
            doc_count = st.session_state.rag_system['collection'].count()
            st.metric("📚 Documents", doc_count)
        except:
            st.metric("📚 Documents", "Unknown")
    else:
        st.warning("🟡 System Not Initialized")
    
    # Sample questions
    st.markdown("## 💡 Sample Questions")
    sample_questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning",
        "What is overfitting?",
        "Difference between supervised and unsupervised learning"
    ]
    
    for question in sample_questions:
        if st.button(f"❓ {question}", key=f"sample_{question}"):
            if st.session_state.initialized:
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
            else:
                st.warning("Please initialize the system first!")

# Main chat interface
if not st.session_state.initialized:
    st.info("👆 Please initialize the RAG system using the sidebar to start chatting!")
    
    # Show project information
    st.markdown("""
    ## 🎯 About This Project
    
    This RAG (Retrieval-Augmented Generation) chatbot provides information about machine learning, 
    deep learning, AI, and related topics using:
    
    - **🤖 Generation Model**: Google Gemini 2.5 Flash
    - **🔗 RAG Framework**: LangChain
    - **🗄️ Vector Database**: Chroma
    - **📚 Dataset**: The Pile (EleutherAI/the_pile) from Hugging Face
    - **🌐 Interface**: Streamlit
    
    ### 🚀 How It Works
    
    1. **Data Loading**: Text data from The Pile dataset is loaded and filtered for ML/AI content
    2. **Embedding**: Text is processed and embedded using sentence transformers
    3. **Storage**: Embeddings are stored in Chroma vector database
    4. **Retrieval**: Relevant context is retrieved for user queries
    5. **Generation**: Gemini generates answers using retrieved context
    
    ### 📝 Sample Questions You Can Ask
    
    - What is machine learning?
    - How do neural networks work?
    - Explain deep learning
    - What is overfitting in ML?
    - Difference between supervised and unsupervised learning
    - What is natural language processing?
    - How does computer vision work?
    - Explain reinforcement learning
    """)
    
else:
    # Chat interface
    st.markdown("## 💬 Chat with the AI Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about ML/AI..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # RAG pipeline
                    rag_system = st.session_state.rag_system
                    
                    response, documents, distances = rag_pipeline(prompt, rag_system)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Show retrieval info
                    with st.expander("🔍 Retrieval Information"):
                        st.write(f"**Retrieved Documents**: {len(documents)}")
                        st.write(f"**Similarity Scores**: {[f'{d:.3f}' for d in distances]}")
                        
                        for i, doc in enumerate(documents):
                            st.write(f"**Document {i+1}**: {doc[:200]}...")
                
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 RAG Chatbot | Powered by Google Gemini 2.5 Flash + LangChain + Chroma</p>
    <p>📚 Knowledge Base: The Pile Dataset (EleutherAI/the_pile)</p>
</div>
""", unsafe_allow_html=True)