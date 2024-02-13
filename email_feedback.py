from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.tools.retriever import create_retriever_tool

load_dotenv()


# Create Retriever
loader = TextLoader("./email_feedback.txt", encoding='UTF-8')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)
embedding = AzureOpenAIEmbeddings()
vectorStore = FAISS.from_documents(docs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})


model = AzureChatOpenAI(
    openai_api_version="2023-07-01-preview",
    azure_deployment="gpt-4-1106",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert and analysing customer feedback from emails. Using the document sources provide useful information such as content summary, sentiment and any other useful information to provide insight for the users asking the questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


retriever_tool = create_retriever_tool(
    retriever,
    "customer_email_feedback_analysis",
    "Use this tool to analyse customer emails on feedback for Widgets"
)
tools = [retriever_tool]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    return response["output"]


def predict(message, history):
    chat_history = []

    response = process_chat(agentExecutor, message, chat_history)
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=response))

    return response

gr.ChatInterface(
    predict,
    chatbot=gr.Chatbot(height=800),
    title="Widget Feedback Analyser").launch()