# PDF Query: A Multimedia RAG System

## 📝 Summary
PDF Query is a full-stack web application that implements a Retrieval-Augmented Generation (RAG) pipeline using a combination of text and image data. The system allows users to upload PDF documents and then ask questions about the content.  

The backend processes these PDFs, extracts both text and images, and creates embeddings for efficient retrieval. Using a Large Language Model (LLM), the application provides accurate, context-aware answers to user queries, referencing both textual and visual information from the documents.  

The project showcases key concepts in modern AI, data processing, and web development.

---

## 💡 Key Features
- **User Authentication**: Secure user registration and login with hashed passwords.  
- **PDF Processing**: Extracts both text and images from uploaded PDF files.  
- **Multimedia RAG**: Creates embeddings for both text and images to support multi-modal querying.  
- **Intelligent Answering**: Uses a Gemini-powered LLM to generate responses based on retrieved context.  
- **Scalable Architecture**: Containerized backend using Docker and a separate frontend deployed on Vercel.  

---

## 🛠️ Technologies Used

### Backend
- **FastAPI**: A modern, high-performance web framework for the API.  
- **LangChain & LangGraph**: Orchestration frameworks for building the RAG pipeline.  
- **Google Gemini**: The Large Language Model (LLM) used for generating responses.  
- **ChromaDB**: A lightweight vector database for storing and retrieving document and image embeddings.  
- **PyMuPDF**: For efficient PDF parsing and data extraction.  
- **PyMongo**: The official MongoDB driver for Python, used for user management.  
- **Bcrypt**: A library for securely hashing user passwords.  
- **Docker**: For containerizing the backend application, ensuring portability and consistent deployment.
- **Azure Web App**: The platform used for deploying the backend container.

### Frontend
- **React**: A popular JavaScript library for building the user interface.  
- **TypeScript**: A typed superset of JavaScript for enhanced code quality.  
- **Shadcn UI**: A collection of reusable components for a modern, accessible user interface.  
- **Vercel**: The platform used for deploying the frontend.  

---

## ✅ Strengths & ❌ Weaknesses

### ✅ Strengths
- **Multi-modal RAG**: Successfully integrates both text and image processing, enabling richer and more accurate query responses.  
- **Robust Core Logic**: Leverages LangChain and ChromaDB to build a solid foundation for the RAG pipeline.  
- **Modern Deployment**: Uses Docker for containerization and Vercel for frontend deployment, reflecting production-ready practices.  
- **Secure Authentication**: Implements password hashing with bcrypt to ensure user credentials are protected.  

### ⚠️ Current Limitations & Next Steps
- **Session Management**: Currently using a basic token system; plan to improve security with JWT-based session handling.  
- **Code Quality Improvements**: Some parts of the code (e.g., hardcoded paths and broad `try...except` blocks) will be refactored for maintainability and cleaner debugging.  
- **Scalability**: While Docker provides portability, the database and component initialization need optimization for handling higher traffic loads in the future.  

