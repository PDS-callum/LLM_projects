import operator
from typing import Annotated, Any, List, Literal, NotRequired, TypedDict
from ..models.llms import llm

from pydantic import BaseModel, Field

class State(TypedDict):
    topic: str
    optimised_search_query: str
    research_items: Annotated[List[Any], operator.add]
    answer: str
    review: Literal["Sufficient","Elaborate"]
    research_calls: NotRequired[int]  # number of times we've run search (persisted so loop can increase results)

class ResearchReview(BaseModel):
    review_result: Literal["Sufficient","Elaborate"] = Field(
        description=(
            "When provided with an answer to a question it must be deemed either sufficient or elaborate."
            "Sufficient - The answer fully answers the question and no further detail is needed."
            "Elabroate - The answer does not fully answer the question and needs further research."
        )
    )
reviewer_llm = llm.with_structured_output(ResearchReview)