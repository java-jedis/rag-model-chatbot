import os
import json
from typing import List, Dict, Any
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-pro"
VOLUMES = ["volume_i", "volume_ii"]

class RAGChatbot:
    def __init__(self):
        # Initialize embeddings (using HuggingFace embeddings)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )
            
        self.vector_store = None
        self.chat_history = []

    def load_json_data(self) -> List[Document]:
        """Load JSON data from volumes and convert to documents."""
        documents = []
        
        for volume in VOLUMES:
            for root, _, files in os.walk(volume):
                for file in files:
                    if file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r') as f:
                                data = json.load(f)
                                # Convert JSON to text
                                text = self._json_to_text(data)
                                # Create document
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        "source": filepath,
                                        "volume": volume
                                    }
                                )
                                documents.append(doc)
                        except Exception as e:
                            print(f"Error processing {filepath}: {e}")
        
        return documents

    def _json_to_text(self, json_obj: Any) -> str:
        """Convert JSON object to text format for embedding."""
        if isinstance(json_obj, dict):
            return "\n".join([f"{k}: {self._json_to_text(v)}" for k, v in json_obj.items()])
        elif isinstance(json_obj, list):
            return "\n".join([self._json_to_text(item) for item in json_obj])
        else:
            return str(json_obj)

    def create_vector_store(self):
        """Create vector store from documents."""
        # Load documents
        documents = self.load_json_data()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

    def query(self, question: str) -> Dict:
        """Process a query and return response."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(question, k=5)
        
        # Format context for the LLM
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided context.
        
        Context information:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer the question based only on the provided context. If you don't know the answer, say that you don't know.
        """)
        
        # Format the chat history
        chat_history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in self.chat_history])
        
        # Create the formatted prompt
        formatted_prompt = prompt.format(
            context=context,
            chat_history=chat_history_str,
            question=question
        )
        
        # Get response from LLM
        response = self.llm.invoke(formatted_prompt)
        
        # Add to chat history
        self.chat_history.append((question, response.content))
        
        # Return response and source documents
        return {
            "answer": response.content,
            "source_documents": docs
        }

# Create Streamlit UI
def main():
    st.title("RAG Chatbot with Gemini")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            chatbot = RAGChatbot()
            chatbot.create_vector_store()
            st.session_state.chatbot = chatbot
            st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    if user_query := st.chat_input("Ask a question:"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.query(user_query)
                st.write(response["answer"])
                
                # Show sources
                with st.expander("Sources"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.write(f"Source {i+1}: {doc.metadata['source']}")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
