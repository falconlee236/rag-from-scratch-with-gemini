import dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain import hub

def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain, retriever):
    """RAG on each sub-question"""
    # Use our decomposition
    sub_questions = sub_question_generator_chain.invoke({"question": question})
    
    # Initialize a list to hold RAG chain results
    rag_results = []
    for sub_question in sub_questions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.invoke(sub_question)
        # Use retrieved documents and sub-question in RAG chain
        answer = (
            prompt_rag
            | ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            | StrOutputParser()
        ).invoke({"context": retrieved_docs, "question": sub_question})
        
        rag_results.append(answer)
    
    return rag_results, sub_questions

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start = 1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

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
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(blog_docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    retriever = vectorstore.as_retriever()
    
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output with interrogative sentence (3 queries):""" # gemini가 하도 의문문으로 질문을 생성 안해서 의문문으로 만들어달라고 추가함
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    
    # decomposition chain
    generate_queries_decomposition = (
        prompt_decomposition
        | ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        | StrOutputParser()
        | (lambda x: x.strip().split("\n"))
    )
    
    # Run
    question = "What are the main components of an LLM-powered autonomous agent system?"
    questions = generate_queries_decomposition.invoke({"question": question})
    print(questions)
    
    """Answer Resursively"""
    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)
    
    q_a_pairs = ""
    for q in questions:
        rag_chain = (
            dict(
                context=itemgetter("question") | retriever,
                question=itemgetter("question"),
                q_a_pairs=itemgetter("q_a_pairs")
            )
            | decomposition_prompt
            | ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke({
            "question": q,
            "q_a_pairs": q_a_pairs
        })
        q_a_pair = f"Question: {q}\nAnswer: {answer}\n\n".strip()
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
    
    print("Answer Resursively")
    print(answer)
    print("---")
    
    """Answer Individually"""
    # RAG prompt
    prompt_rag = hub.pull("rlm/rag-prompt")
    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_decomposition, retriever)
    
    context = format_qa_pairs(questions, answers)
    
    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    final_rag_chain = (
        prompt
        | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        | StrOutputParser()
    )
    
    print("Answer Individually")
    print(final_rag_chain.invoke({"context": context, "question": question}))
    print("----")
    