import bs4
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
def main():
    ''' load Documents '''
    loader = WebBaseLoader(
        web_paths=(
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
        ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=(
                    "post-content",
                    "post-title",
                    "post-header",
                )
            )
        ),
    )
    docs = loader.load()
    
    # split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # Embed
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=VertexAIEmbeddings(model_name="text-embedding-005")
    )
    retriever = vectorstore.as_retriever()
    
    ''' Retrival and generation '''
    
    # Prompt
    prompt = hub.pull('rlm/rag-prompt')
    
    # LLM
    llm = ChatVertexAI(
        model="gemini-1.5-flash-001",
        temperature=0,
	)
    
    rag_chain = (
        dict( # post-processing
            context=retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            question=RunnablePassthrough()
        ) | prompt | llm | StrOutputParser()
    )
    
    # Question
    print(rag_chain.invoke("What is Task Decomposition?"))

if __name__ == "__main__":
    load_dotenv()
    main()


