import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from PIL import Image
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import get_buffer_string
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
from prompts import (
    analyst_instructions,
    question_instructions,
    search_instructions,
    answer_instructions,
    section_writer_instructions,
    report_writer_instructions,
    intro_conclusion_instructions,
)


llm = ChatOpenAI(model="gpt-4o-mini")
# ---------------- Generate Analysts ----------------
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
    
class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: list[Analyst]  


def create_analysts(state: GenerateAnalystsState):
    """ Create analysts """
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    
    structured_llm = llm.with_structured_output(Perspectives)
    
    system_message = analyst_instructions.format(
        topic=topic, 
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )
    
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts.")]
    )
    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass
      
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

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search Query for retrieval")

def generate_question(state: InterviewState):
    """ Node to generate a question """
    
    # Get State
    analyst = state["analyst"]
    messages = state["messages"]
    
    system_messages = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_messages)] + messages)
    
    return {"messages": [question]}

tavily_search = TavilySearch(max_results=3)


# Search query writing
def search_web(state: InterviewState):
    """ Retrieve docs from web search """
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=search_instructions)] + state["messages"])
    
    search_docs = tavily_search.invoke(search_query.search_query)
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs["results"]
        ]
    )
    
    return {"context": [formatted_search_docs]}

def search_wikipedia(state: InterviewState):
    """ Retrieve docs from wikipedia """
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])
    
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

def generate_answer(state: InterviewState):
    """ Node to answer a question """
    
    analyst = state["analyst"]
    context = state["context"]
    messages = state["messages"]
    
    system_messages = answer_instructions.format(goals=analyst.persona, context=context)
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

def write_section(state: InterviewState):
    """ Node to answer a question """
    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
    
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
    
    return {"sections": [section.content]}  


interview_builder = StateGraph(InterviewState, output_schema=InterviewOutputState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges(
    "answer_question",
    route_messages,
    ["ask_question", "save_interview"],
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# memory = MemorySaver()
# graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

# ---------------- Finalize Report ----------------
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

def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """    
    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback is not None:
        return "create_analysts"
    else:
        topic = state["topic"]
        return [
            Send(
                "conduct_interview", {
                        "analyst": analyst, 
                        "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
                    }
                ) for analyst in state["analysts"]
        ]

def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join(f"{s}" for s in sections)
    system_messages = [SystemMessage(
        content=report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    )]
    report = llm.invoke(system_messages + [HumanMessage(content=f"Write a report based upon these memos.")])
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join(f"{s}" for s in sections)
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join(f"{s}" for s in sections)
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
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


builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion",write_conclusion)
builder.add_node("finalize_report",finalize_report)


builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

def main_graph(max_analysts, topic, thread):

    # Run the graph until the first interruption
    for event in graph.stream({"topic":topic,
                            "max_analysts":max_analysts}, 
                            thread, 
                            stream_mode="values"):
        
        analysts = event.get('analysts', '')
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)  

    # # Confirm we are happy
    # feedback = input("Enter feedback to revise analysts, or press Enter to continue: ")
    # graph.update_state(
    #     thread, 
    #     {"human_analyst_feedback": feedback if feedback.strip() else None},
    #     as_node="human_feedback",
    # )

    # # Continue
    # for event in graph.stream(None, thread, stream_mode="updates"):
    #     print("--Node--")
    #     node_name = next(iter(event.keys()))
    #     print(node_name)

    # Interactive feedback loop
    while True:
        feedback = input("Enter feedback to revise analysts, or press Enter to continue: ")
        
        if feedback.strip():
            # Provide feedback and re-enter graph at 'human_feedback'
            graph.update_state(
                thread,
                {"human_analyst_feedback": feedback},
                as_node="human_feedback",
            )

            # Continue the graph until next interruption
            for event in graph.stream(None, thread, stream_mode="values"):
                analysts = event.get("analysts", "")
                if analysts:
                    for analyst in analysts:
                        print(f"Name: {analyst.name}")
                        print(f"Affiliation: {analyst.affiliation}")
                        print(f"Role: {analyst.role}")
                        print(f"Description: {analyst.description}")
                        print("-" * 50)
        else:
            break  # Exit feedback loop

    for event in graph.stream(None, thread, stream_mode="updates"):
        print("--Node--")
        node_name = next(iter(event.keys()))
        print(node_name)

        
    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')
    return report
    
if __name__ == "__main__":
    # Inputs
    max_analysts = 3 
    # topic = "The benefits of adopting LangGraph as an agent framework"
    topic = "The Chinese's NBA player Hansen Yang."
    thread = {"configurable": {"thread_id": "1"}}
    report = main_graph(max_analysts, topic, thread)
    
    # 保存为 Markdown 文件
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)