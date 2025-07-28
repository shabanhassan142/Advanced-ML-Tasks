# Context-Aware Wikipedia Chatbot with Streamlit

## Objective

Build a conversational chatbot that can remember context and retrieve external information from Wikipedia articles using Retrieval-Augmented Generation (RAG) and LangChain, deployed as a web app with Streamlit.

---

## Methodology / Approach

1. **Data Source:**
   - The chatbot imports articles directly from Wikipedia using the `wikipedia` Python library.

2. **Vectorization & Storage:**
   - The imported article is vectorized using `HuggingFaceEmbeddings` (from `langchain_community.embeddings`).
   - The vectorized data is stored in a local vector store using `ChromaDB`.

3. **Conversational Memory:**
   - The chatbot uses `ConversationBufferMemory` from LangChain to remember the chat history and maintain context across multiple user queries.

4. **Retrieval-Augmented Generation (RAG):**
   - When a user asks a question, the chatbot retrieves relevant information from the vector store and generates a context-aware answer using an LLM (OpenAI's GPT, via API key).

5. **Deployment:**
   - The app is deployed as a web application using Streamlit, providing an interactive UI for importing articles and chatting.

---

## How to Run

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your OpenAI API key:**
   - Edit `app.py` and set your key, or set the environment variable `OPENAI_API_KEY`.
3. **Run the App:**
   ```bash
   python -m streamlit run app.py
   ```
4. **Open your browser:**
   - Go to the URL shown in the terminal (usually http://localhost:8501 or http://localhost:8502).

---

## Key Results or Observations

- **Seamless Wikipedia Integration:**
  - Users can import any Wikipedia article by topic and immediately start asking questions about its content.

- **Contextual Q&A:**
  - The chatbot remembers previous questions and answers, enabling more natural, context-aware conversations.

- **Retrieval-Augmented Generation:**
  - Answers are generated using both the imported article and conversational context, improving relevance and accuracy.

- **Easy Deployment:**
  - The app runs locally via Streamlit and requires only a browser to interact with.

- **Extensible:**
  - The approach can be extended to other document sources or LLMs with minimal changes.

---

## Example Use Case

1. Import the article "Artificial intelligence" from Wikipedia.
2. Ask: "What is AI?"
3. Follow up: "Who are some pioneers in this field?"
4. The chatbot will answer using the imported article and remember the context of your conversation.

---

## Troubleshooting
- If you see errors about missing modules, install them with `pip install <module-name>`.
- If the app does not open, ensure you are running the command in the correct directory and your virtual environment is activated.
- For network errors (e.g., downloading models), ensure you have a stable internet connection.

---

## Credits
- Built with [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Streamlit](https://streamlit.io/), and [Wikipedia](https://pypi.org/project/wikipedia/). 