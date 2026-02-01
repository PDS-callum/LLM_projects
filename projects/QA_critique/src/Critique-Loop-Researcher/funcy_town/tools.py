import time
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from ..models.llms import llm

@tool
def get_search_data(search_query: str, max_items: int):
    """Search the web for information using the given query."""
    last_error = None
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(1.5 * attempt)  # backoff on retry
            results = DDGS().text(search_query, max_results=max_items)
            if results:
                return results
        except Exception as e:
            last_error = e
    if last_error:
        raise last_error
    return []

tools = [get_search_data]
tools_by_name = {t.name:t for t in tools}
researcher_llm = llm.bind_tools([get_search_data])