import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from ..models.llms import llm
from ..funcy_town.tools import get_search_data
from ..objs.classes import State, reviewer_llm

logger = logging.getLogger(__name__)


def search_query_optimiser(state: State):
    logger.info("→ Search query optimiser: optimising query for topic '%s'", state["topic"])
    result = llm.invoke(
        [
            SystemMessage(content="""You are a search query optimiser. Your ONLY job is to output a single, short search engine query (e.g. 3–8 words, keyword-style) that would best find answers to the user's question.

Rules:
- Output NOTHING except the search query. No explanation, no "Search for:", no quotes, no preamble.
- Use keywords and short phrases, like someone would type into Google or DuckDuckGo.
- One query only. Example: for "tallest mountain on earth" you might output: tallest mountain world"""),
            HumanMessage(content=f"Research topic: {state['topic']}\n\nOutput only the search query:"),
        ]
    )
    logger.info("← Search query optimiser: query = '%s'", result.content[:80] + "..." if len(result.content) > 80 else result.content)
    return {"optimised_search_query": result.content}


def get_search_results(state: State):
    query = state.get("optimised_search_query", "") or state.get("topic", "")
    logger.info("→ Get search results: running search (query: '%s')", query[:60])
    # Call the search tool directly so we always get real results (the LLM with
    # bound tools would return tool_calls that we weren't executing before).
    research_calls = state.get("research_calls", 0)
    max_items = 15 + research_calls * 5
    raw = get_search_data.invoke({"search_query": query, "max_items": max_items})
    if not raw:
        logger.warning("← Get search results: 0 results from search")
        content = f"(No search results returned for: {query})"
    else:
        # DDGS.text() returns list of dicts with title, href, body
        parts = []
        for i, item in enumerate(raw, 1):
            if isinstance(item, dict):
                title = item.get("title", "")
                body = item.get("body", "")
                parts.append(f"[{i}] {title}\n{body}")
            else:
                parts.append(str(item))
        content = "\n\n".join(parts)
        logger.info("← Get search results: got %d result(s), %d chars", len(raw), len(content))
    return {
        "research_items": [content],
        "research_calls": state.get("research_calls", 0) + 1
    }


def compile_answer(state: State):
    logger.info("→ Compile answer: synthesising answer for '%s'", state["topic"])
    topic = state["topic"]
    research_results = state["research_items"]
    result = llm.invoke([
        SystemMessage(content="Your job is to compile research into an answer to a question."),
        HumanMessage(content=f"""The original question was '{state['topic']}'. 
        The research available is {state['research_items']}.
        Please generate an answer to the question based on the research.""")
    ])
    logger.info("← Compile answer: answer ready (%d chars)", len(result.content))
    print("\n" + "─" * 60 + "\nAnswer:\n" + "─" * 60 + "\n" + result.content + "\n" + "─" * 60 + "\n")
    return {"answer": result.content}


def research_reviewer(state: State):
    logger.info("→ Research reviewer: checking if answer is sufficient")
    result = reviewer_llm.invoke(
        [
            SystemMessage(content="""You are a reviewer who's sole purpose is to determine whether an initial question has been answered in sufficient detail. Please only reply either sufficient or elaborate depending on whether the answer is sufficient.

Rules:
- If the question has been answered and is correct you will return 'Sufficient"
- If the question is unanswered or the answer is wrong you will return 'Elaborate"
            """),
            HumanMessage(content=f"My initial question was '{state['topic']}' and the answer I have recieved is '{state['answer']}'. Has my question be answered sufficiently?")
        ]
    )
    logger.info("← Research reviewer: %s", result.review_result)
    return {"review": result.review_result}


def should_continue(state: State):
    if state["review"] == "Elaborate":
        logger.info("→ Should continue: Elaborate → looping back to search")
        return "GSR"
    else:
        logger.info("→ Should continue: Sufficient → finishing")
        return END