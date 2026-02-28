from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain
from dotenv import load_dotenv

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

#define generation node
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

#define reflection node
def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"message": [HumanMessage(content=res.content)]}

#tell graph what the state will be and how to update it
builder = StateGraph(state_schema=MessageGraph)

#add the nodes to the graph
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)


#accepts state and output of function will be a string representing the node name
#this function will rune everytime we run the generation node
#output will telegraph where to go next.  
#either reflection node or to the end
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    
    return REFLECT


#define edges
builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END,REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)


#compile the graph
graph = builder.compile()
#print(graph.get_graph().draw_mermaid())


def main():
    print("Starting langgraph-example2!")
    
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }

    #invoke graph with the input
    #hold response in response variable
    response = graph.invoke(inputs)

    #response is a dictionary that holds list of AI messages
    #we want to print the last AI message
    last_message = response["messages"][-1]
    print(last_message.content)

if __name__ == "__main__":
    main()

