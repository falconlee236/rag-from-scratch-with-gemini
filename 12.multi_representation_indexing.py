import dotenv
import uuid
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

if __name__ == "__main__":
    dotenv.load_dotenv()
    
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()
    
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
    docs.extend(loader.load())
    print(len(docs)) # 2
    chain = (
		{"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, max_retries=0)
        | StrOutputParser()
	)
    
    summaries = chain.batch(docs, {"max_concurrency": 5})
    
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    
    # The storage Layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"
    
    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    
    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]
    
    # Add
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    
    query = "Memory in agents"
    sub_docs = vectorstore.similarity_search(query, k=1)
    print(sub_docs[0])
    
    retrieved_docs = retriever.invoke(query, n_results=1)
    print(retrieved_docs[0].page_content[:500])