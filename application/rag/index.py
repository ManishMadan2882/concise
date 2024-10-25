from htmlTemplate import css, bot_template, user_template  # type: ignore
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup

load_dotenv()

llmtemplate = """[INST]
You are a research assistant, please help to answer questions from the context provided
{question}
[/INST]
"""


def load_data(inputs):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    urls = inputs if isinstance(inputs, list) else [inputs]
    documents = []

    for url in urls:
        try:
            loader = WebBaseLoader([url], header_template=headers)
            fetched_docs = loader.load()  # Fetch content from the web
            for doc in fetched_docs:
                # Add title and content for web documents
                documents.append({
                    "title": f"Web content from {url}",
                    "content": doc.page_content
                })
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            continue

    return documents

# Prepare method for both PDFs and web content
def prepare_docs(pdf_docs, urls=None):
    """Process the uploaded or generated PDF files and URLs."""
    docs = []
    metadata = []
    content = []

    # Process PDF documents
    if pdf_docs:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for index, page in enumerate(pdf_reader.pages):
                doc_page = {
                    "title": f"{pdf.name if hasattr(pdf, 'name') else 'Remote PDF'} page {index + 1}",
                    "content": page.extract_text(),
                }
                docs.append(doc_page)
    
    # Process URLs and fetch web content
    if urls:
        web_documents = load_data(urls)
        docs.extend(web_documents)

    # Extract content and metadata
    for doc in docs:
        content.append(doc["content"])
        metadata.append({"title": doc["title"]})
    
    return content,metadata


def prepare_docs_from_urls(url_list):
    docs = []
    metadata = []
    content = []
    documents = load_data(url_list)
    print(documents)
    for url in url_list:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.title.string if soup.title else url
            text_content = soup.get_text()

            doc_page = {"title": title, "content": text_content}
            docs.append(doc_page)
        else:
            print(f"Failed to retrieve {url}")

    for doc in docs:
        content.append(doc["content"])
        metadata.append({"title": doc["title"]})

    return content, metadata


def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Split documents into {len(split_docs)} passages")
    return split_docs


def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = "vectorstore/db_faiss"
    db.save_local(DB_FAISS_PATH)
    return db


def get_conversation_chain(vectordb):
    llama_llm = ChatOllama(model="llama3.2",temperature=0)

    retriever = vectordb.as_retriever()
    ##CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(llmtemplate)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llama_llm,
        retriever=retriever,
        # condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        memory=memory,
        return_source_documents=True,
    )
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain


def validate_answer_against_sources(response_answer, source_documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    similarity_threshold = 0.5
    source_texts = [doc.page_content for doc in source_documents]

    answer_embedding = model.encode(response_answer, convert_to_tensor=True)
    source_embeddings = model.encode(source_texts, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)

    if any(score.item() > similarity_threshold for score in cosine_scores[0]):
        return True

    return False


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        print(i)
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            print(message.content)
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.sidebar.title("URL Input")
        num_urls = st.sidebar.number_input(
            "Number of URLs", min_value=1, max_value=10, value=1
        )
        urls = []
        for i in range(num_urls):
            url = st.sidebar.text_input(f"Enter URL {i + 1}")
            if url:
                urls.append(url)
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                content, metadata = prepare_docs(pdf_docs,urls)
                split_docs = get_text_chunks(content, metadata)

                # create vector store
                vectorstore = ingest_into_vectordb(split_docs)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
