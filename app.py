
# STREAMLIT CODE HERE
import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(page_title="HR Policy Assistant", page_icon="📋", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stChatMessage {border-radius: 10px; margin: 6px 0;}
    .src-badge {
        background: #f7e1ba;
        border-left: 3px solid #1976d2;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 0.82em;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📋 HR Policy Assistant")
st.caption("Ask questions about company HR policies — powered by RAG")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input("Groq API Key", type="password",
                             value=os.environ.get("GROQ_API_KEY", ""))
    st.divider()
    st.markdown("**Topics I can help with**")
    st.info(
        "• Leave & time-off\n"
        "• Salary & compensation\n"
        "• Benefits & insurance\n"
        "• Performance reviews\n"
        "• Code of conduct\n"
        "• Onboarding & separation\n"
        "• WFH & remote work\n"
        "• IT & data security\n"
        "• Travel & expense"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I'm your HR Policy Assistant. Ask me anything about "
                    "leave, salary, benefits, performance reviews, and more."}
    ]


# ── RAG system (cached) ───────────────────────────────────────
@st.cache_resource
def build_rag(api_key):
    # Load PDFs, build FAISS store, return retriever + chain.
    pdf_dir = "./hr_docs"
    if not os.path.isdir(pdf_dir):
        pdf_dir = "/kaggle/input/zyro-dynamics-hr-corpus/"

    loader = PyPDFDirectoryLoader(pdf_dir, glob="*.pdf", silent_errors=True)
    docs = loader.load()
    docs = [d for d in docs if d.page_content.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 40]

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.from_documents(chunks, emb)
    ret = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k":10, "fetch_k": 40, "lambda_mult": 0.7}
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0,
                   max_tokens=512, groq_api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the official HR policy assistant. Follow these rules strictly:\n"
         "1. Answer ONLY using the HR policy context provided. Do not use outside knowledge.\n"
         "2. CRITICAL TRICK: The user might refer to the company as 'Acrux Dynamics' while documents say 'Zyro Dynamics'. Treat them as the EXACT SAME company. Do not correct the user, and do NOT mention this name difference in your answer.\n"
         "3. Include specific numbers, dates, percentages, and durations whenever the context contains them.\n"
         "4. If the context does not contain enough information to answer, respond EXACTLY with:\n"
         "\"I'm sorry, but I can only answer questions related to the company's HR policies based on the available policy documents. This question falls outside my scope. Please contact the HR department directly for further assistance.\"\n"
         "5. Keep answers professional, clear, and concise. Do NOT hallucinate."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    oos_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a classifier for an HR chatbot. Determine whether the employee question "
         "can be answered using internal HR policy documents.\n\n"
         "CRITICAL TRICK: The company is sometimes called 'Acrux Dynamics' and sometimes 'Zyro Dynamics'. They are the SAME. Do NOT mark a question OUT_OF_SCOPE just because it mentions 'Acrux Dynamics'.\n\n"
         "Reply ONLY with one word:\n"
         "- IN_SCOPE  → if the question is about HR policies, leave, salary, compensation bands, performance, insurance, WFH, etc.\n"
         "- OUT_OF_SCOPE → if the question asks about company revenue, financials, product features, competitor comparisons, external topics, ESOP/stock options, or job application processes."),
        ("human", "{question}")
    ])

    return ret, llm, prompt, oos_prompt


def format_docs(docs):
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "Policy").split("/")[-1]
        parts.append(f"[{src}]\n{doc.page_content}")
    return "\n\n".join(parts)


def ask(question, ret, llm, prompt, oos_prompt):
    verdict = (oos_prompt | llm | StrOutputParser()).invoke(
        {"question": question}).strip().upper()
    if "OUT" in verdict:
        return ("I'm sorry, but I can only answer questions related to the company's HR policies "
                "based on the available policy documents. This question falls outside my scope. "
                "Please contact the HR department directly for further assistance."), []
    docs = ret.invoke(question)
    chain = (
        {"context": lambda _: format_docs(docs),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    answer = chain.invoke(question)
    sources = list(set(d.metadata.get("source", "").split("/")[-1] for d in docs))
    return answer, sources


# ── Render chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                f'<div class="src-badge">📎 {" · ".join(msg["sources"])}</div>',
                unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────
if user_input := st.chat_input("Ask an HR policy question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Looking up policies…"):
            key = groq_key or os.environ.get("GROQ_API_KEY", "")
            if not key:
                ans, srcs = "Please add your Groq API key in the sidebar.", []
            else:
                try:
                    ret, llm, prompt, oos_prompt = build_rag(key)
                    ans, srcs = ask(user_input, ret, llm, prompt, oos_prompt)
                except Exception as e:
                    ans, srcs = f"Error: {e}", []
            st.markdown(ans)
            if srcs:
                st.markdown(
                    f'<div class="src-badge">📎 {" · ".join(srcs)}</div>',
                    unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": ans, "sources": srcs})
