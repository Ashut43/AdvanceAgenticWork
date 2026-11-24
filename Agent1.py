#First Import all libraries Related to the Agents
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition,tool_validator
from langchain_community.tools import DuckDuckGoSearchRun,QuerySQLDatabaseTool 
from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from typing import List,Annotated,Dict,TypedDict,Literal,Any
from pydantic import Field,BaseModel
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from pathlib import Path;import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
load_dotenv()
class toolhistory(BaseModel):
      tool_name:str
      tool_input:Dict[str,Any]
      tool_output:Any
class evaluate(BaseModel):
    score:Annotated[float,Field(description="score from 1 to 10",le=10,ge=1)]
    Feedback:Literal['Positive','Negative']
class state(TypedDict):
    question:str=Field(description="user questuion based on anything")
    answer:str=Field(description='Anser according to the Question')
    EvaluateData:evaluate
    tool:toolhistory
model=ChatOpenAI(
    model="qwen3.8-27b",api_key=os.getenv('api_key'),
    base_url=os.getenv('base_url'),temperature=0.5,
    max_tokens=500,)
llm=ChatOpenAI(
    model="qwen3.8-27b",api_key=os.getenv('api_key'),
    base_url=os.getenv('base_url'),temperature=0.7,
    max_tokens=800,)
#customtools
#@tool — Function-based tools.
#BaseTool — Class-based tools.
#FunctionTools
@tool
def Translator(text:str,Targetl:str,SourceL:str):
    """ this is a good translator you provided both language it can return translated DDcument"""
    Translator=GoogleTranslator(
          source=SourceL,
          target=Targetl
    )
    TranslatedText=Translator.translate(text)
    return TranslatedText
@tool
def Examinertool(Permission:bool,state:Annotated[state,InjectedState])->str:
    """Check Answer if not appropriate to the Question than search on Internet and make Answer and return it"""
    Question=state['question']
    if state['EvaluateData'].score<5:
              if Permission in ['aprooved','passed','yes','go for this']:
                     return "Thankyou sir Have a Nice day bsc you are Enginner"
              elif Permission in ['update','not good','very bad']:
                      serch=DuckDuckGoSearchRun.invoke('Question')
                      prompt=PromptTemplate(
                            template="""you are Good Answer Generator from InternetSearched,
                            please generate  perfect answer to the: {question}from provide data:{search}""",
                            input_variables=['question','search']
                      )
                      chain=prompt|llm
                      Answer=chain.invoke({'question':Question,'search':search})
                      return Answer
    else:
          return f"you are Great bros"

tools=[Examinertool,Translator]
Tool_map={
      tool.name:tool
      for tool in tools
}
model=model.bind_tools(tools)
def Generator(state:state):
      Question=state['question']
      prompt=PromptTemplate(
                template="""
                         you are good Answer Generator take
                         :{Question}and gives the eject Answer related Given Answer")
                       """,
                  input_variables=['Question']
      )
      chain=prompt|model
      Answer=chain.invoke({'Question':Question})
      return {'Answer':Answer}
def Evaluator(state:state):
      Question=state['question'] 
      Answer=state['answer']
      prompt=PromptTemplate(
            template="""
              you are good Evaluator
              human',"please Evaluate Answer:{Answer} based on Question:{Question}and gives the eject Answer related output structure")
       """,
       input_variables=['Question','Answer']
       ) 
      chain1=prompt|model.with_structured_output(evaluate)
      result=chain1.invoke({'Answer':Answer,"Question":Question})
      return {'EvaluateData':result}
def ToolNode(state:state):
      tool=state['tool']
      toolinput=tool.tool_input
      toolOutput=tool.tool_output
      tool_name=tool.tool_name
      selected_tool=Tool_map.get(tool_name)
      if selected_tool is None:
            result="tool is not found"
            return result
      result=selected_tool.invoke(toolinput)
      updated_tool=tool.model_copy(update={'tool_output':result})
      return {'tool':updated_tool}
def tool_condition(state:state):
    """it tells route to the model what do if tool has been called"""
    tool_name=state['tool'].tool_name
    if tool_name in Tool_map:
        return tool_name
graph=StateGraph(state)

graph.add_node('Generator',Generator)
graph.add_node('Evaluator',Evaluator)
graph.add_node('tools',ToolNode)

graph.add_edge(START,'Generator')
graph.add_edge('Generator','Evaluator')
graph.add_conditional_edges('Evaluator',tools_condition,{'tools':'tools','end':END})
graph.add_edge('tools',END)

Agent=graph.compile()
result=Agent.invoke({'question':"what is the differance between llm and Transformer "})
print('result')





