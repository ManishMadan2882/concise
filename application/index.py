import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

load_dotenv()

class RAG:
    def load(self): ##pdf file address
        loader = PyPDFLoader(self.pdf_file)
        self.pages = loader.load()
        print("Loading PDF")
        
    def split_into_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunks = splitter.split_documents(self.pages)
        print(f"Number of chunks: {len(chunks)}")
        print(f"Length of a chunk: {len(chunks[1].page_content)}")
        print("Content of a chunk:", chunks[1].page_content)
        self.chunks = chunks
    
    def set_vectorstore(self):
        embeddings = OllamaEmbeddings(model=os.getenv('MODEL'))
        vectorstore = FAISS.from_documents(self.chunks, embeddings)
        self.vectorstore = vectorstore
    
    def set_retriever(self):
        self.retriever = self.vectorstore.as_retriever()
    
    def load_model(self):
        self.model = ChatOllama(model=os.getenv('MODEL'), temperature=1)
    
    def parse_response(self):
        self.parser = StrOutputParser()
        self.chain = self.model | self.parser 
        
    def prompt_template(self):
        template = """
        You are a research assistant that provides answers to queries based on
        a given context. 
        Answer the question based on the context. If you can't answer the
        question, reply "I don't know".

         Be as concise as possible and go straight to the point.

        Context: {context}
        Question: {question}    
         """
        self.prompt = PromptTemplate.from_template(template)
    
    def create_chain(self):
        from operator import itemgetter

        self.chain = (
        {
        "context": itemgetter("question") | self.retriever,
        "question": itemgetter("question"),
        }
        | self.prompt
        | self.model
        | self.parser
        )
    
    def stream(self):
        for question in questions:
             print(f"Question: {question}")
             print(f"Answer: {self.chain.invoke({'question': question})}")
             print("*************************\n")
    
    def __init__(self, pdf_file,questions):
        self.pdf_file = pdf_file
        self.questions = questions
questions = [
            "Summarise the content mentioned in this research work",
            "Explain the working methodology proposed by researcher",
            "Conclude the outcomes of the provided research work"
        ] 
rag = RAG(pdf_file='phishing.pdf',questions=questions)

rag.load()
rag.split_into_chunks()
rag.set_vectorstore()
rag.set_retriever()
rag.load_model()
rag.parse_response()
rag.prompt_template()
rag.create_chain()
rag.stream()