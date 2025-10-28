# 🤖 InsightRAG Chatbot: ML/AI Knowledge Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u4hwe39XZZlQbtdQDecR4MStnQlgBj2h?usp=sharing)

A fully functional Retrieval-Augmented Generation (RAG) chatbot that provides comprehensive information about machine learning, deep learning, AI, and related topics. Built with modern AI technologies and ready for deployment.

## 🎯 Project Purpose

This RAG chatbot serves as an intelligent knowledge assistant specializing in machine learning, deep learning, and artificial intelligence topics. The chatbot leverages a sophisticated retrieval-augmented generation pipeline to provide accurate, contextual answers by combining:

- **Knowledge Retrieval**: Accessing relevant information from a curated ML/AI knowledge base
- **Contextual Generation**: Using Google Gemini 2.5 Flash to generate comprehensive responses
- **Interactive Learning**: Enabling users to explore complex AI concepts through natural conversation

The primary goal is to make AI and machine learning knowledge accessible through an intuitive, conversational interface that can handle both basic concepts and advanced technical questions.

## 📚 Dataset Information

🧩 Dataset Summary

[arXiv:2302.14035](https://arxiv.org/abs/2302.14035)

The Pile is a large-scale, diverse, open-source text dataset developed by EleutherAI to train and evaluate large language models (LLMs).
It contains over 800GB of English text collected from 22 high-quality sources, including academic publications, web pages, GitHub repositories, and books.

The dataset was designed to provide a broad and representative sample of human language, covering domains such as:

Artificial Intelligence and Machine Learning research papers

Mathematics, Science, and Engineering texts

Open web content and Wikipedia articles

GitHub code snippets and documentation

Due to its diversity and scale, The Pile has become a benchmark dataset for developing and testing modern language models.


📊 Data Source

This project uses The Pile
 — an open-source text dataset created by EleutherAI and hosted on Hugging Face.
The dataset consists of diverse English text collected from various open-access sources, including academic papers, web pages, and books.

Data is accessed directly via the Hugging Face API using the datasets library, without downloading or storing local copies.

📚 Reference

For academic and citation purposes, The Pile dataset is introduced in the following research paper:

Gao, L., Tow, J., Biderman, S., Black, S., Anthony, Q., Golding, L., ... & Leahy, C. (2020). The Pile: An 800GB Dataset of Diverse Text for Language Modeling. arXiv preprint arXiv:2302.14035

This dataset supports the development and evaluation of lightweight LLM experiments for educational and research purposes.


### Dataset 

- **Primary Dataset**: The Pile (EleutherAI/the_pile) from Hugging Face
- **Access Method**: Hugging Face Datasets API (no local downloads required)
- **Content Type**: Text-only data (no tables, images, or PDFs)

### Dataset Structure

The dataset contains diverse text content filtered specifically for ML/AI relevance:

- **Content Filtering**: Text samples are filtered using ML/AI keywords including:

  - Machine learning, deep learning, neural networks
  - Artificial intelligence, algorithms, models
  - Training, data, features, classification
  - Regression, clustering, optimization, gradient, tensor

- **Text Processing**:

  - Content is cleaned and preprocessed
  - Text is chunked into manageable pieces (500 words with 50-word overlap)
  - Only substantial chunks (100-2000 characters) are retained
  - Text is embedded using sentence transformers for vector search

- **Storage**: Processed text chunks are stored in Chroma vector database for efficient similarity search

### Usage in RAG Pipeline

The dataset serves as the knowledge base for the RAG system, enabling:

- Semantic search for relevant context
- Contextual answer generation
- Comprehensive coverage of ML/AI topics

## 🔧 Methods Used

### RAG Pipeline Architecture

The chatbot implements a sophisticated Retrieval-Augmented Generation pipeline:

#### 1. **Data Processing Pipeline**

```
Raw Text → Filtering → Chunking → Embedding → Vector Storage
```

- **Text Filtering**: ML/AI keyword-based content selection
- **Chunking**: Intelligent text segmentation with overlap
- **Embedding**: Sentence transformer-based vectorization
- **Storage**: Chroma vector database for efficient retrieval

#### 2. **Retrieval System**

- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Vector Database**: Chroma with persistent storage
- **Similarity Search**: Cosine similarity for document retrieval
- **Context Assembly**: Top-k relevant documents combined

#### 3. **Generation System**

- **Language Model**: Google Gemini 2.5 Flash
- **Temperature**: 0.7 for balanced creativity and accuracy
- **Context Integration**: Retrieved documents used as context
- **Response Formatting**: Markdown support for rich text

#### 4. **Technical Stack**

- **RAG Framework**: LangChain for pipeline orchestration
- **Vector Database**: Chroma for embedding storage and retrieval
- **Embeddings**: Sentence Transformers for text vectorization
- **LLM**: Google Gemini 2.5 Flash for response generation
- **Interface**: Streamlit for web-based chat interface

## 📊 Results Summary

The RAG chatbot successfully provides comprehensive answers across multiple ML/AI domains:

### **Answer Quality**

- **Contextual Accuracy**: Responses are grounded in retrieved knowledge
- **Comprehensive Coverage**: Handles both basic and advanced topics
- **Structured Output**: Well-formatted responses with examples
- **Technical Depth**: Can explain complex algorithms and concepts

### **Performance Metrics**

- **Response Time**: Fast retrieval and generation (< 5 seconds)
- **Relevance**: High-quality context retrieval from knowledge base
- **Coverage**: Extensive ML/AI topic coverage
- **Usability**: Intuitive conversational interface

### **Capabilities Demonstrated**

- Explains fundamental ML/AI concepts
- Provides algorithm explanations with examples
- Offers practical implementation guidance
- Covers current trends and advanced topics
- Handles both theoretical and applied questions

## 💡 Example Questions

The chatbot can answer a comprehensive range of questions across multiple categories:

### **Basic Concepts**

- What is the difference between AI, machine learning, and deep learning?
- Can you explain supervised, unsupervised, and reinforcement learning?
- What are features and labels in a dataset?
- Explain overfitting vs underfitting.

### **Algorithms & Models**

- How does a neural network learn?
- What is gradient descent and how does it work?
- Explain decision trees and random forests.
- What are convolutional neural networks (CNNs) used for?
- How does a transformer model like GPT work?

### **Practical Applications**

- How to preprocess data for machine learning?
- How can I use AI for image recognition?
- Give an example of AI in healthcare.
- What are common pitfalls when training deep learning models?

### **Technical Details**

- What is backpropagation?
- How does regularization prevent overfitting?
- Explain embedding vectors and similarity search.
- What are activation functions and why are they important?

### **Performance & Optimization**

- How to improve model accuracy?
- What is cross-validation and why is it used?
- Explain hyperparameter tuning.
- What is transfer learning?

### **Trends & Advanced**

- Explain reinforcement learning with examples.
- What are large language models and how do they work?
- How is generative AI different from predictive AI?
- What is the future of AI in finance/medicine?

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u4hwe39XZZlQbtdQDecR4MStnQlgBj2h?usp=sharing)

1. **Open the notebook**: Click the Colab badge above or upload `InsightRAG.ipynb` to Google Colab
2. **Set up API key**: Add your Gemini API key to Colab secrets
3. **Run all cells**: Execute the notebook to build the RAG system
4. **Test the system**: Try the sample questions provided

### Option 2: Local Development

1. **Create virtual environment**:

   ```bash
   python -m venv rag_chatbot_env
   source rag_chatbot_env/bin/activate  # On Windows: rag_chatbot_env\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**:

   ```bash
   export GOOGLE_API_KEY="your_gemini_api_key_here"
   ```

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

5. **Access the interface**: Open `http://localhost:8501` in your browser

## 🔑 API Key Setup

### Google Colab

1. Go to the key icon (🔑) in the left sidebar
2. Add a new secret with key `GEMINI_API_KEY` and your API key as value
3. Restart the runtime and run the notebook

### Local/Hugging Face Spaces

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set it as an environment variable: `GOOGLE_API_KEY`
3. Or enter it directly in the Streamlit interface

## 🏗️ Solution Architecture

### Problem Statement

Traditional chatbots often provide generic responses without access to specific domain knowledge. This project solves the challenge of creating an AI assistant that can provide accurate, contextual information about machine learning and AI topics.

### Technology Stack

- **Frontend**: Streamlit for web interface
- **Backend**: Python with LangChain framework
- **Vector Database**: Chroma for embedding storage
- **Embeddings**: Sentence Transformers
- **LLM**: Google Gemini 2.5 Flash
- **Data Source**: The Pile dataset via Hugging Face

### Architecture Benefits

- **Scalable**: Can handle multiple users simultaneously
- **Accurate**: Grounded responses using retrieved context
- **Flexible**: Easy to extend with additional knowledge sources
- **Efficient**: Fast retrieval and generation pipeline

## 🌐 Web Interface 

### Local Testing

1. Run `streamlit run app.py`
2. Open `http://localhost:8501`
3. Enter your Gemini API key
4. Initialize the RAG system
5. Start chatting!


### Interface Features

- **Chat Interface**: Clean, responsive design
- **Real-time Responses**: Instant AI-generated answers
- **Context Display**: Shows retrieved documents and similarity scores
- **Sample Questions**: Quick-start buttons for common queries
- **System Status**: Real-time monitoring of RAG system health

## 📁 Project Structure

```
Chatbot_Project/
├── rag_notebook.ipynb      # Complete Colab notebook with RAG pipeline
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
└── chroma_db/             # Vector database (created during execution)
```

## 🔧 Configuration Options

The system can be customized through various parameters:

```python
# RAG Pipeline Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GEMINI_MODEL = 'gemini-2.0-flash-exp'
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024
N_RETRIEVAL_RESULTS = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Gemini API key is correctly set
2. **Memory Issues**: Reduce the number of documents processed in Colab
3. **Chroma Connection**: Check if the vector database directory exists
4. **Model Loading**: Ensure all dependencies are installed correctly

### Solutions

- **Restart Runtime**: In Colab, use Runtime → Restart Runtime
- **Check Logs**: Look for error messages in the console
- **Verify Dependencies**: Run `pip list` to check installed packages
- **Test Components**: Use the test functions in the notebook

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **EleutherAI** for The Pile dataset
- **Google** for Gemini API
- **LangChain** for RAG framework
- **Chroma** for vector database
- **Streamlit** for web interface
- **Hugging Face** for dataset access and deployment platform

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the notebook comments and documentation
3. Open an issue in the repository
4. Contact the development team

---

**🚀 Ready to explore the world of AI with our RAG chatbot!**

_Built with ❤️ using modern AI technologies_
