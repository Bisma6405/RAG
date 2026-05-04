import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import tempfile
import traceback
from sentence_transformers import SentenceTransformer
import time

# Load Env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM1_MODEL = os.getenv("LLM1_MODEL", "llama-3.3-70b-versatile")
LLM2_MODEL = os.getenv("LLM2_MODEL", "mixtral-8x7b-32768")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found!")
    st.stop()

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False


def get_embeddings():
    """Get embeddings model with error handling"""
    try:
        # Use a smaller, faster model
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    except Exception as e:
        st.error(f"❌ Error loading embeddings: {str(e)}")
        return None


def process_pdf(uploaded_file):
    """Process PDF with detailed progress"""
    try:
        # Step 1: Save file
        st.write("💾 Saving file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Step 2: Load PDF
        st.write("📖 Loading PDF...")
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No content found in PDF")

        st.write(f"✓ Loaded {len(docs)} pages")

        # Step 3: Split text
        st.write("✂️ Splitting text...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for faster processing
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        st.write(f"✓ Created {len(chunks)} chunks")

        # Step 4: Create embeddings (with progress)
        st.write("🧮 Creating embeddings (this may take 1-2 minutes)...")
        embeddings = get_embeddings()

        if embeddings is None:
            raise ValueError("Failed to load embeddings")

        # Create vectorstore with progress
        st.write(" Processing chunks...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=None  # In-memory only
        )

        # Cleanup temp file
        os.unlink(tmp_path)

        st.write("✅ PDF processed successfully!")
        return vectorstore, len(chunks)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error(f"📋 {traceback.format_exc()}")
        return None, 0


def generate_response(question, history, retriever, llm1, llm2):
    """Generate response with fallback"""
    try:
        # Retrieve context
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant context found."

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful AI assistant. Answer using the Context and Chat History."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ])

        chain1 = prompt | llm1 | StrOutputParser()
        chain2 = prompt | llm2 | StrOutputParser()
        inputs = {"history": history}

        # Try LLM1
        try:
            res = chain1.invoke(inputs)
            if res and len(res.strip()) > 10:
                return res
            raise ValueError("Response too short")
        except Exception as e1:
            st.warning(f"⚠️ Primary model failed, trying backup...")
            try:
                return chain2.invoke(inputs)
            except Exception as e2:
                return f"⛔ Both models failed: {str(e2)[:200]}"

    except Exception as e:
        return f"⛔ Error: {str(e)}"


# ================= STREAMLIT UI =================
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 AI PDF Chatbot")

with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.get("current_file") != uploaded_file.name:
            st.info(f"📄 {uploaded_file.name}")

            # Clear previous state
            st.session_state.vectorstore = None
            st.session_state.retriever = None

            # Process with progress
            vectorstore, chunk_count = process_pdf(uploaded_file)

            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.session_state.current_file = uploaded_file.name
                st.session_state.file_processed = True
                st.session_state.messages = []
                st.success(f"✅ Ready! ({chunk_count} chunks)")
                st.rerun()
            else:
                st.session_state.file_processed = False
                st.error("❌ Processing failed")
        else:
            st.success("✅ PDF loaded")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.file_processed:
    st.info("👆 Upload a PDF to start")
    st.stop()

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]

            llm1 = ChatGroq(model=LLM1_MODEL, api_key=GROQ_API_KEY, temperature=0.3)
            llm2 = ChatGroq(model=LLM2_MODEL, api_key=GROQ_API_KEY, temperature=0.3)

            response = generate_response(prompt, history, st.session_state.retriever, llm1, llm2)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()