import bs4
import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

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
    
    text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50
    )
    splits = text_spliter.split_documents(blog_docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    retriever = vectorstore.as_retriever()
    
    # Few Shot Examples
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )
    
    generate_queries_step_back = (
        prompt 
        | ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        | StrOutputParser()
    )
    question = "What is task decomposition for LLM agents?"
    print(generate_queries_step_back.invoke({"question": question}))
    
    # Response prompt
    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    
    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back | retriever,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        | StrOutputParser()
    )
    print(chain.invoke({"question": question}))