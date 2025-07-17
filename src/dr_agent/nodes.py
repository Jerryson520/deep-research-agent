from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph, MessagesState
from dr_agent.state import (
    InterviewOutputState,
    InterviewState,
    ResearchGraphState,
    Analyst,
    Perspectives,
)
from dr_agent.configuration import Configuration
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.constants import Send
from langgraph.types import Command
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_tavily import TavilySearch
from langchain_core.messages import get_buffer_string
from typing import Literal, Optional
from dr_agent import prompts
from langgraph.types import interrupt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_llm(config: Optional[RunnableConfig] = None):
    configuration = Configuration.from_runnable_config(config)
    # Extract just the model name from the "provider/model-name" format
    model_parts = configuration.model.split('/')
    model_name = model_parts[-1] if len(model_parts) > 1 else configuration.model
    llm = ChatOpenAI(model=model_name)
    return llm

# ---------------- Generate Analysts ----------------
def create_analysts(state: ResearchGraphState, config: Optional[RunnableConfig] = None):
    """ Create analysts """
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(Perspectives)
    
    system_message = prompts.ANALYST_INSTRUCTIONS.format(
        topic=topic, 
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )
    
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts.")]
    )
    return {"analysts": analysts.analysts}

def confirm_analysts(state: ResearchGraphState):
    """Interrupt to show generated analysts and ask user for feedback."""
    analysts = state["analysts"]
    
    return interrupt({
        "message": "Here is the list of generated analysts. Press enter to confirm or provide feedback to regenerate.",
        "generated_analysts": analysts
    })

def initiate_all_interviews(state: ResearchGraphState) -> Command[Literal["create_analysts", "conduct_interview"]]:
    """ This is the "map" step where we run each interview sub-graph using Send API """    
    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback is not None:
        return Command(update={}, goto="create_analysts")
    else:
        topic = state["topic"]
        sends = [
            Send(
                "conduct_interview", {
                        "analyst": analyst, 
                        "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
                    }
                ) for analyst in state["analysts"]
        ]

        return Command(update={"human_analyst_feedback": human_analyst_feedback}, goto=sends)

# ---------------- Conduct Interviews ----------------  
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search Query for retrieval")

def generate_question(state: InterviewState, config: Optional[RunnableConfig] = None):
    """ Node to generate a question """
    
    # Get State
    analyst = state["analyst"]
    messages = state["messages"]
    
    llm = get_llm(config)
    system_messages = prompts.QUESTION_INSTRUCTIONS.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_messages)] + messages)
    
    return {"messages": [question]}

# Initialize TavilySearch after environment variables are loaded
tavily_search = TavilySearch(max_results=3)


# Search query writing
def search_web(state: InterviewState, config: Optional[RunnableConfig] = None):
    """ Retrieve docs from web search """
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=prompts.SEARCH_INSTRUCTIONS)] + state["messages"])
    
    search_docs = tavily_search.invoke({"query": search_query.search_query})
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs["results"]
        ]
    )
    
    return {"context": [formatted_search_docs]}

def search_wikipedia(state: InterviewState, config: Optional[RunnableConfig] = None):
    """ Retrieve docs from wikipedia """
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=prompts.SEARCH_INSTRUCTIONS)] + state["messages"])
    
    search_docs = WikipediaLoader(
        query=search_query.search_query,
        load_max_docs=2,
    ).load()
    
    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    
    return {"context": [formatted_search_docs]}

def generate_answer(state: InterviewState, config: Optional[RunnableConfig] = None):
    """ Node to answer a question """
    
    analyst = state["analyst"]
    context = state["context"]
    messages = state["messages"]
    
    llm = get_llm(config)
    system_messages = prompts.ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_messages)] + messages)
    
    answer.name = "expert"
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    """ Save interviews """
    
    messages = state["messages"]
    interview = get_buffer_string(messages)
    
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    """ Route between question and answer """
    
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)
    
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    
    if num_responses >= max_num_turns:
        return "save_interview"
    
    last_messages = messages[-2]
    if "Thank you so much for your help" in last_messages:
        return "save_interview"
    else:
        return "ask_question"

def write_section(state: InterviewState, config: Optional[RunnableConfig] = None):
    """ Node to answer a question """
    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
    
    llm = get_llm(config)
    system_message = prompts.SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
    
    return {"sections": [section.content]}

# ---------------- Finalize Report ----------------
def write_report(state: ResearchGraphState, config: Optional[RunnableConfig] = None):
    sections = state["sections"]
    topic = state["topic"]
    
    llm = get_llm(config)
    formatted_str_sections = "\n\n".join(f"{s}" for s in sections)
    system_messages = [SystemMessage(
        content=prompts.REPORT_WRITER_INSTRUCTIONS.format(topic=topic, context=formatted_str_sections)
    )]
    report = llm.invoke(system_messages + [HumanMessage(content=f"Write a report based upon these memos.")])
    return {"content": report.content}

def write_introduction(state: ResearchGraphState, config: Optional[RunnableConfig] = None):
    sections = state["sections"]
    topic = state["topic"]
    
    llm = get_llm(config)
    formatted_str_sections = "\n\n".join(f"{s}" for s in sections)
    
    # Summarize the sections into a final report
    
    instructions = prompts.INTRO_CONCLUSION_INSTRUCTIONS.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = llm.invoke([SystemMessage(content=instructions)]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState, config: Optional[RunnableConfig] = None):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    llm = get_llm(config)
    # Concat all sections together
    formatted_str_sections = "\n\n".join(f"{s}" for s in sections)
    
    # Summarize the sections into a final report
    
    instructions = prompts.INTRO_CONCLUSION_INSTRUCTIONS.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = llm.invoke([SystemMessage(content=instructions)]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None
        
    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}
