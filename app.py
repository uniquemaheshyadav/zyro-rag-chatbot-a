import streamlit as st
import os
import time
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

st.set_page_config(page_title="HR Policy Assistant", page_icon="📋", layout="wide")

st.markdown('''
<style>
    .main {background-color: #f5f7fa;}
    .stChatMessage {border-radius: 10px; margin: 6px 0;}
    .src-badge {
        background: #e3f2fd; border-left: 3px solid #1976d2;
        padding: 6px 10px; border-radius: 4px;
        font-size: 0.82em; margin-top: 6px;
    }
</style>
''', unsafe_allow_html=True)

st.title("📋 HR Policy Assistant")
st.caption("Ask questions about company HR policies — powered by RAG")
st.divider()

with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    st.divider()
    st.info("Topics: Leave, Salary, WFH, Performance, Insurance, Conduct.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about HR policies."}]

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."

@st.cache_resource
def build_rag(api_key):
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
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.from_documents(chunks, emb)
    ret = vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.85})

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=512, groq_api_key=api_key)

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are the official HR policy assistant. Answer ONLY using the HR policy context provided.\n"
         "IMPORTANT: Documents may mention 'Zyro Dynamics' or 'Acrux Dynamics' — treat them as the SAME company.\n"
         "Rules:\n"
         "1. Answer ONLY using information explicitly present in the context.\n"
         "2. Include exact numbers, dates, percentages, durations, and ALL eligibility conditions (e.g., minimum days worked) exactly as they appear. NEVER omit conditions or caveats.\n"
         "3. If context has PARTIAL information, give that information directly — no refusal preamble.\n"
         "4. NEVER hallucinate. NEVER ask which company — just answer.\n"
         "5. CRITICAL — Leave types: Each leave type (Earned, Sick, Maternity, etc.) has DIFFERENT rules. Answer ONLY for the specific leave type asked. NEVER mix rules between leave types.\n"
         "6. CRITICAL — Insurance types: If asked about 'health insurance' or 'medical insurance', answer ONLY about Group Medical Insurance. Ensure you mention the coverage amount of Rs. 5,00,000 per year, and that it covers employee + spouse + up to 2 dependent children, as stated in the policy documents.\n"
         "7. CRITICAL — Complete lists and timelines: Include EVERY item in the context. NEVER give a partial list. If context has a 7-row APR table, give all 7 rows. If context has 4 WFH types, give all 4 types. Do NOT include internal process-ownership columns (e.g., 'Owner', 'Department') unless the question specifically asks who is responsible.\n"
         "8. No repetition: state each fact ONCE; do not restate the conclusion again at the end.\n"
         "9. CRITICAL — Style: Write in a DIRECT, FACTUAL, policy-document tone — like a sentence "
         "taken straight from an HR policy manual. Do NOT use conversational framing such as "
         "'At Acrux Dynamics,' 'According to the policy,' or repeating the company name. "
         "Do not add extra context the question did not ask for. State the fact(s) plainly "
         "and concisely, as the source document itself would state them."),
        ("human", "HR POLICY CONTEXT:\n{context}\n\nEMPLOYEE QUESTION:\n{question}")
    ])

    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a classifier for an HR chatbot. Determine whether the employee question can be answered using internal HR policy documents.\n\n"
         "CRITICAL: 'Acrux Dynamics' and 'Zyro Dynamics' are the SAME company. Do NOT mark OUT_OF_SCOPE just because it mentions 'Acrux Dynamics'.\n\n"
         "Reply ONLY with one word:\n"
         "- IN_SCOPE → if the question is about HR policies, leave, salary, compensation, performance, insurance, WFH, onboarding, separation, travel, conduct, IT security.\n"
         "- OUT_OF_SCOPE → ONLY if the question asks about company revenue, financials, product features, competitor comparisons, ESOP/stock options, job applications, or recruitment process."),
        ("human", "{question}")
    ])

    return ret, llm, RAG_PROMPT, OOS_PROMPT

def format_docs(docs):
    formatted_parts = []
    for doc in docs:
        source_name = doc.metadata.get("source", "HR Policy").split("/")[-1]
        formatted_parts.append(f"--- Source: {source_name} ---\n{doc.page_content}")
    return "\n\n".join(formatted_parts)

def _invoke_with_retry(chain, inputs, max_retries=5):
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                time.sleep(10 * (attempt + 1))
            else:
                raise e
    raise Exception("Max retries exceeded due to rate limiting.")

def ask(question, ret, llm, prompt, oos_prompt):
    oos_chain = oos_prompt | llm | StrOutputParser()
    verdict = _invoke_with_retry(oos_chain, {"question": question}).strip().upper()
    time.sleep(2)
    
    if "OUT" in verdict:
        return REFUSAL_MESSAGE, []
    
    docs = ret.invoke(question)
    rag_chain = (
        {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    answer = _invoke_with_retry(rag_chain, question)
    time.sleep(2)
    
    sources = list(set(d.metadata.get("source", "").split("/")[-1] for d in docs))
    return answer.strip(), sources

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(f'<div class="src-badge">📎 {" · ".join(msg["sources"])}</div>', unsafe_allow_html=True)

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
                st.markdown(f'<div class="src-badge">📎 {" · ".join(srcs)}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
