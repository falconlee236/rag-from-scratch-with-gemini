import tiktoken
import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name=encoding_name)
    num_tokens = len(encoding.encode(text=string))
    print(encoding.encode(text=string))
    return num_tokens

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

if __name__ == "__main__":
    dotenv.load_dotenv()
    # Documents
    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."
    
    # use tiktoken to count number of tokens
    print("count token used by tiktoken")
    print(num_tokens_from_string(question, "cl100k_base"))
    
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    query_result = embedding.embed_query(question)
    document_result = embedding.embed_query(document)
    print("get embedding result Google Gen AI Embedding Models")
    print(len(query_result))
    
    # get similaity with query and document
    print("Cosine similarity: ", cosine_similarity(query_result, document_result))
    
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
    print(blog_docs)
    
    # split
    text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50
    )
    
    # make splits
    splits = text_spliter.split_documents(blog_docs)
    for x in splits:
        print(x, end="\n********\n\n")
        
    # Vectorstores
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )
    
    retriever = vectorstore.as_retriever()
    print(vectorstore)