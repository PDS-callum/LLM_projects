import logging
from langgraph.graph import StateGraph, START, END
from .funcy_town.nodes import search_query_optimiser, get_search_results, research_reviewer, should_continue, compile_answer
from .objs.classes import State

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    logger = logging.getLogger(__name__)
    logger.info("Building graph...")
    builder = StateGraph(State)

    builder.add_node("SQO", search_query_optimiser)
    builder.add_node("GSR", get_search_results)
    builder.add_node("CA", compile_answer)
    builder.add_node("RR", research_reviewer)

    builder.add_edge(START, "SQO")
    builder.add_edge("SQO", "GSR")
    builder.add_edge("GSR", "CA")
    builder.add_edge("CA", "RR")
    builder.add_conditional_edges(
        "RR",
        should_continue,
        ["GSR", END]
    )

    agent = builder.compile()
    logger.info("Graph compiled. Starting research...")

    # Invoke
    topic = "In quantum mechanics, how do electrons travel between orbitals through a zone of probability density 0?"
    logger.info("Topic: %s", topic)
    result = agent.invoke({"topic": topic})

    logger.info("Done.")
    print("Answer:", result.get("answer", "N/A"))


if __name__ == "__main__":
    main()