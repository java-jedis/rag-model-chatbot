import os
import json
import re
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-pro"
VOLUMES = ["volume_i", "volume_ii"]

class JSONKnowledgeBaseTool(BaseTool):
    name = "json_knowledge_base"
    description = "Search for information in the JSON knowledge base"
    vector_store = None
    
    def _run(self, query: str) -> str:
        if not self.vector_store:
            return "Knowledge base not initialized"
        
        docs = self.vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata['source'] for doc in docs]
        
        return f"Based on the knowledge base:\n{context}\n\nSources: {', '.join(sources)}"
    
    def _arun(self, query: str) -> str:
        return self._run(query)

class AgenticRAGChatbot:
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
        self.knowledge_tool = JSONKnowledgeBaseTool()
        
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

    def _extract_metadata(self, json_obj: Dict) -> Dict:
        """Extract metadata from JSON object."""
        metadata = {}
        
        # Extract common metadata fields if they exist
        for field in ['id', 'title', 'date', 'author', 'category', 'type']:
            if field in json_obj:
                metadata[field] = json_obj[field]
        
        return metadata

    def initialize(self):
        """Initialize the agentic RAG chatbot."""
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
        
        # Set up tools
        self.knowledge_tool.vector_store = self.vector_store

    def query(self, question: str) -> str:
        """Process a query and return response."""
        if not self.vector_store:
            return "Agent not initialized. Call initialize() first."
        
        try:
            # Get context information using the knowledge tool
            context = self.knowledge_tool._run(question)
            
            # Create a prompt for the LLM
            prompt = f"""
            You are an agentic RAG chatbot that helps answer questions about legal JSON data.
            
            Previous conversation:
            {self._format_chat_history()}
            
            User question: {question}
            
            {context}
            
            Based on the provided context, answer the user's question in a helpful and informative way.
            If the answer is not in the context, say that you don't have that information.
            """
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Add to chat history
            self.chat_history.append((question, response.content))
            
            return response.content
            
        except Exception as e:
            return f"Error processing your request: {str(e)}"
            
    def _format_chat_history(self):
        """Format chat history for prompt context."""
        if not self.chat_history:
            return "No previous conversation."
            
        formatted = []
        for i, (q, a) in enumerate(self.chat_history[-5:]):  # Get last 5 exchanges
            formatted.append(f"User: {q}")
            formatted.append(f"Assistant: {a}")
            
        return "\n".join(formatted)

# Create Streamlit UI
def main():
    st.title("Agentic RAG Chatbot with Gemini")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading knowledge base and initializing agent..."):
            chatbot = AgenticRAGChatbot()
            chatbot.initialize()
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
                st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
