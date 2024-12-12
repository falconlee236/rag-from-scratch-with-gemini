import tiktoken
import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name=encoding_name)
    num_tokens = len(encoding.encode(text=string))
    print(encoding.encode(text=string))
    return num_tokens

if __name__ == "__main__":
    dotenv.load_dotenv()
    # Documents
    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."
    
    # use tiktoken to count number of tokens
    print("count token used by tiktoken")
    print(num_tokens_from_string(question, "cl100k_base"))
    
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_result = embedding.embed_query(question)
    document_result = embedding.embed_query(document)
    print("get embedding result Google Gen AI Embedding Models")
    print(query_result)
    print(len(query_result))
    
    