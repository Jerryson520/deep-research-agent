import operator
from langgraph.graph import START, END, StateGraph, MessagesState
from typing import List, Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# ---------------- Analyst Models ----------------
class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst"
    )
    name: str = Field(
        description="Name of the analyst"
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic"
    )
    description: str = Field(
        description="Description of the analyst focus, concerns and motives."
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n" 
    
    
class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations"
    )

# ---------------- Conduct Interviews ----------------    
class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list

class InterviewOutputState(MessagesState):
    max_num_turns: int
    sections: list

# ---------------- Research Graph State ----------------
class ResearchGraphInputState(TypedDict):
    topic: str
    max_analysts: int

class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: list[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str

class ResearchGraphOutputState(TypedDict):
    topic: str
    analysts: list[Analyst]
    final_report: str