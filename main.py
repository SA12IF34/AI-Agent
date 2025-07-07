import os
from pathlib import Path
import sqlite3

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

import environ


ENV = environ.Env()

environ.Env.read_env(os.path.join(Path(__file__).resolve(), '.env'))

os.environ['DEEPSEEK_API_KEY'] = ENV('DEEPSEEK_API_KEY')
os.environ['TAVILY_API_KEY'] = ENV('TAVILY_API_KEY')

tavily_search = TavilySearchResults(max_results=3)
@tool
def web_search(input_: str) -> dict | str:
    """
    Search the web for desired thing
    """

    results = tavily_search.invoke(input_)
    return results


tools = [web_search]

class State(TypedDict):

    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatDeepSeek(
    model='deepseek-chat',
    temperature=1.5,
    max_retries=3
)
llm = llm.bind_tools(tools)



def chatbot(state: State):

    message = llm.invoke(state['messages'])
    assert len(message.tool_calls) <= 1
    return {'messages': [message]}

tool_node = ToolNode(tools=tools)

# memory = MemorySaver()

graph_builder.add_node('tools', tool_node)
graph_builder.add_node('chatbot', chatbot)

graph_builder.add_edge(START, 'chatbot')
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition
)
graph_builder.add_edge('tools', 'chatbot')


conn = sqlite3.connect('checkpoint.sqlite3', check_same_thread=False)
memory = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=memory)

message = 'hello, how are you?'

output = graph.invoke({
        'messages': [
            {'role': 'system', 'content': '''
                You are a helpful assistant, you can search the web.
                You must follow the rules delimited by backticks.
                The rules: ```
                - Do not use emojis in your responses
                - Respond with the same language the user used to talk to you
                - Make your responses five sentences at most
                - If you are asked to search the web, use web_earch tool, extract the urls from it's output, and format your response as JSON with the following key:
                    urls: <the list of urls extracted from web_search tool output>
                ```
            '''},
            {'role': 'human', 'content': message}
        ]
    }, config={'configurable': {'thread_id': '1'}}, stream_mode='values')['messages'][-1].content

print(output)