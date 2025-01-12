from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
import datetime
import dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""
    '''
    Let’s assume we’ve built an index that:

    1. Allows us to perform unstructured search over the contents and title of each document
    2. And to use range filtering on view count, publication date, and length.
    We want to convert natural langugae into structured search queries.

    We can define a schema for structured search queries.
    '''

    contents_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts"
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )

    def pretty_print(self) -> None:
        for field in self.model_fields:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.model_fields[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")



if __name__ == "__main__":
    dotenv.load_dotenv()
    docs = YoutubeLoaderDL.from_youtube_url(
        "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
    ).load()
    print(docs)
    print(docs[0].metadata)

    system = """You are an expert at converting user questions into database queries. \
    You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
    Given a question, return a database query optimized to retrieve the most relevant results.

    If there are acronyms or words you are not familiar with, do not try to rephrase them."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
    structured_llm = llm.with_structured_output(TutorialSearch)
    query_analyzer = prompt | structured_llm
    # llm에게 질문이 주어지면 해당 질문을 스키마 형태로 구조화하는 방법 -> 왜냐? 동영상의 메타데이터를 활용하기 위해서
    print()
    query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()
    print()
    query_analyzer.invoke({"question": "videos on chat langchain published in 2023"}).pretty_print()
    print()
    query_analyzer.invoke({"question": "videos that are focused on the topic of chat langchain that are published before 2024"}).pretty_print()
    print()
    query_analyzer.invoke({"question": "how to use multi-modal models in an agent, only videos under 5 minutes"}).pretty_print()
    # To then connect this to various vectorstores, you can follow https://python.langchain.com/docs/how_to/self_query/