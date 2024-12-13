import dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader

if __name__=="__main__":
    dotenv.load_dotenv()
    
    '''indexing'''
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
    
    