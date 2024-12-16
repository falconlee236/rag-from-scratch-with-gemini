import dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter

def get_unique_union(documents: list[list]) -> list:
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flatten_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flatten_docs))
    return [loads(doc) for doc in unique_docs]


if __name__=="__main__":
    dotenv.load_dotenv()
    
    '''indexing'''
    # Load Blog
    loader = WebBaseLoader(
		web_paths=(
			"https://lilianweng.github.io/posts/2023-06-23-agent/",
		),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
	)
    blog_docs = loader.load()

    # Split
    text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50
    )

    # make splits
    splits = text_spliter.split_documents(blog_docs)

    # index
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    retriever = vectorstore.as_retriever()
    
    '''Prompt'''
    # Multi Query -> Use Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives 
        | ChatGoogleGenerativeAI( model="gemini-1.5-flash", temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n\n")) # langsmith를 까보니까 계속 빈 question이 나와서 개행을 2개 단위로 쪼갬
    )

    # Retrieve
    question = "What is task decomposition for LLM agents?"
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question": question})
    for doc in docs:
        print("----------------")
        print(doc)
        print("----------------")

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI( model="gemini-1.5-pro", temperature=0)
    final_rag_chain = (
        dict(
            context=retrieval_chain,
            question=itemgetter("question")
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    print(final_rag_chain.invoke({"question": question}))

