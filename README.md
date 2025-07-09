# RAG Chatbot with JSON Data

This project implements a Retrieval-Augmented Generation (RAG) chatbot that uses legal JSON data to provide intelligent responses. The chatbot uses Google Gemini AI model and HuggingFace embeddings for text representation.

## Project Structure

```
rag-model-chatbot/
├── .env                  # API keys and configuration
├── requirements.txt      # Python dependencies
├── volume_i/             # JSON data folder 1
├── volume_ii/            # JSON data folder 2
└── src/
    ├── explore_data.py           # Script to explore JSON data structure
    ├── rag_chatbot.py            # Basic RAG chatbot implementation
    └── agentic_rag_chatbot.py    # Advanced agentic RAG chatbot
```

## Setup Instructions
1. Create and activate a venv file first
    ```bash
    python3 -m venv venv
   source venv/bin/activate

   ```


2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key in the `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. Explore your JSON data structure:
   ```bash
   python3 src/explore_data.py
   ```

5. Run the basic RAG chatbot:
   ```bash
   streamlit run src/rag_chatbot.py
   ```

6. Or run the agentic version:
   ```bash
   streamlit run src/agentic_rag_chatbot.py
   ```

