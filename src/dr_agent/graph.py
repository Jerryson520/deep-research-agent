"""Graph definition for the deep research agent."""

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.types import Command
from typing import Literal

from dr_agent.state import (
    InterviewOutputState,
    InterviewState,
    ResearchGraphState,
    ResearchGraphInputState,
    ResearchGraphOutputState,
)
from dr_agent.nodes import (
    # Analyst creation
    create_analysts,
    confirm_analysts,
    initiate_all_interviews,
    
    # Interview nodes
    generate_question,
    search_web,
    search_wikipedia,
    generate_answer,
    save_interview,
    route_messages,
    write_section,
    
    # Report writing nodes
    write_report,
    write_introduction,
    write_conclusion,
    finalize_report,
)


def create_interview_graph():
    """Create the interview subgraph for conducting individual analyst interviews."""
    interview_builder = StateGraph(InterviewState, output_schema=InterviewOutputState)
    
    # Add nodes
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)

    # Add edges
    interview_builder.add_edge(START, "ask_question")
    
    # Parallel search paths
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    
    # Both search paths lead to answer generation
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    
    # Conditional routing after answer
    interview_builder.add_conditional_edges(
        "answer_question",
        route_messages,
        ["ask_question", "save_interview"],
    )
    
    # Final steps
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)
    
    return interview_builder.compile()


def create_research_graph():
    """Create the main research graph that orchestrates the entire research process."""
    builder = StateGraph(
        ResearchGraphState, 
        input_schema=ResearchGraphInputState, 
        output_schema=ResearchGraphOutputState)
    
    # Add nodes
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("confirm_analysts", confirm_analysts) 
    builder.add_node("initiate_all_interviews", initiate_all_interviews)
    builder.add_node("conduct_interview", create_interview_graph())
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    # Add edges
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "confirm_analysts")
    builder.add_edge("confirm_analysts", "initiate_all_interviews")
    
    # After interviews, write report components in parallel
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    
    # All report components must complete before finalization
    builder.add_edge("write_conclusion", "finalize_report")
    builder.add_edge("write_report", "finalize_report")
    builder.add_edge("write_introduction", "finalize_report")
    builder.add_edge("finalize_report", END)

    # Compile with memory and interrupts
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory, # interrupt_before=['initiate_all_interviews'], 
    )
    
    return graph


# Create the main graph instance
research_graph = create_research_graph()

# For backward compatibility, also expose the individual functions
def interview_graph():
    """Backward compatibility function for interview graph creation."""
    return create_interview_graph()

def graph():
    """Backward compatibility function for main graph creation."""
    return create_research_graph()

if __name__ == "__main__":
    # If this file is run directly, print the graph structure
    # print(research_graph.to_dict())
    # print("Research graph created successfully.")
    # Run the graph until the first interruption
    # Inputs
    
    max_analysts = 3 
    # topic = "The benefits of adopting LangGraph as an agent framework"
    topic = "The Chinese's NBA player Hansen Yang."
    thread = {"configurable": {"thread_id": "1"}}
    graph = graph()
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
            # graph.update_state(
            #     thread,
            #     {"human_analyst_feedback": feedback},
            #     as_node="human_feedback",
            # )
            graph.invoke({"human_feedback": feedback}, config=thread)

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
    
    # 保存为 Markdown 文件
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)