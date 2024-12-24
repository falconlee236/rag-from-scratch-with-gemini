import dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.utils.math import cosine_similarity

# Data model
class RouteQuery(BaseModel):
    """ Route a user query to the most relevant datasource """
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
		..., # Ellipsis is often used as a placeholder when the exact content doesn't matter
		description="Given a user question choose which datasource would be most relevant for answering their question"
	)

def choose_route(result: RouteQuery) -> str:
    if "python_docs" in result.datasource.lower():
        ### logic here
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### logic here
        return "chain for js_docs"
    else:
        ### logic here
        return "golang_docs"
    

dotenv.load_dotenv()
# Two prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.
Here is a question:
{query}"""
math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.
Here is a question:
{query}"""

# Embed prompts
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


# Route question to prompt
def prompt_router(input: dict):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity

    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)

if __name__ == "__main__":
    # LLM with function call
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)
    
    # Prompt
    system = """You are an expert at routing a user question to the appropriate data source.

	Based on the programming language the question is referring to, route it to the relevant data source."""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}")
        ]
    )
    
    # define Router
    router = prompt | structured_llm
    
    question = """Why doesn't the following code work:

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
    prompt.invoke("french")
    """
    result = router.invoke({"question": question})
    print(result)
    print(result.datasource)
    
    # logical routing
    full_chain = router | RunnableLambda(choose_route)
    print(full_chain.invoke({"question": question}))
    
    chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | llm
        | StrOutputParser()
    )
    print(chain.invoke("What's a black hole?"))
    
    