# GCP RAG Chatbot

This was originally pieced together using publically available information. Maybe someday I will relase the guide I wrote on how to set up the infrastructure once it is common knowledge. 

A modular Retrieval-Augmented Generation (RAG) chatbot built on Google Cloud Platform (GCP). This project provides tools for document ingestion, embedding, and chat interaction using a custom agent backed by Google Cloud services.

## 🧱 Project Structure

```
gcp-rag-chatbot-main/
├── gcs_chat_app/       # Streamlit chat interface
├── gcs_embed/          # Document embedding pipeline
├── rag_agent/          # RAG agent logic and utilities
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Components

### gcs_chat_app
- Serves chat requests (likely with FastAPI)
- Imports documents from GCS
- Dockerized with Cloud Build support

### gcs_embed
- Converts documents into vector embeddings
- Cloud Build and Workflows-ready
- Integrates with GCP storage and possibly Vertex AI

### rag_agent
- Core RAG agent logic
- Tools for managing conversation state and node logic

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.9+
- Docker
- Google Cloud SDK (`gcloud`)
- Enabled GCP services: Cloud Storage, Artifact Registry, Vertex AI (if used)

### Clone the Repository
```bash
git clone https://github.com/your-org/gcp-rag-chatbot.git
cd gcp-rag-chatbot
```

### Environment Variables
Copy `.env.example` in `gcs_chat_app/` and `gcs_embed/` to `.env` and configure as needed:
```bash
cp gcs_chat_app/.env.example gcs_chat_app/.env
cp gcs_embed/.env.example gcs_embed/.env
```

### Build and Deploy with Cloud Build (Optional)
```bash
gcloud builds submit --config=gcs_chat_app/cloudbuild.yaml
```

Or for embeddings:
```bash
gcloud builds submit --config=gcs_embed/cloudbuild.yaml
```

## 🧠 How it Works
1. Documents are uploaded or fetched from GCS.
2. `gcs_embed` embeds these documents and stores vector data.
3. `gcs_chat_app` accepts chat input, calls the RAG agent, and returns responses.
4. `rag_agent` uses tools and embeddings to retrieve and answer questions.

## 🧪 Local Development
To run the chat app locally:
```bash
cd gcs_chat_app
uvicorn app:app --reload
```

To test embeddings:
```bash
cd gcs_embed
python embed.py
```

## TODO
- Find new open source data source for training materials and them to the repo
- Modify embeddings functionality to fit new data source
- Share my guide on how to set up the infrastructure (Once it is common knowledge)

## 📄 License
MIT License. See the [LICENSE](LICENSE) file for details.

---

This project is designed to be production-ready on GCP and easily extensible with new document sources, models, or UI interfaces. Contributions welcome!
