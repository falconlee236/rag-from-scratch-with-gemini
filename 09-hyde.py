import bs4
import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

if __name__ == "__main__":
    dotenv.load_dotenv()

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

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    splits = text_spliter.split_documents(blog_docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    retriever = vectorstore.as_retriever()

    # HyDE document generation
    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval = (
        prompt_hyde 
        | ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
        | StrOutputParser()
    )

    # Run
    question = "What is task decomposition for LLM agents?"
    print(generate_docs_for_retrieval.invoke({"question": question}))

    # Retrieve
    retrieval_chain = generate_docs_for_retrieval | retriever
    retrieval_docs = retrieval_chain.invoke({"question": question})
    print(retrieval_docs)

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
        | StrOutputParser()
    )

    print("\n---answer--\n")
    print(final_rag_chain.invoke({"question": question, "context": retrieval_docs}))