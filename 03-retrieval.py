import bs4
import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if __name__=="__main__":
    dotenv.load_dotenv()
    # load blog
    loader = WebBaseLoader(
        web_paths=(
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()
    
    # split
    text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50
    )
    
    # make splits
    splits = text_spliter.split_documents(blog_docs)
        
    # Vectorstores - indexing
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs=dict(
            k=1, # 이러면 가장 가까운 1개만 가져옴
        )
    )
    
    docs = retriever.get_relevant_documents("What is Task Decomposition?")
    print(len(docs))
    print(docs)
    
    