from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.environ['GROQ_API_KEY']

## Web Search Agent
we_bsearch_agent = Agent(
    name='Web Search Agent',
    role='Search the web for information',
    model=Groq(id="llama-3.3-70b-versatile",api_key=GROQ_API_KEY),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,
)

## Financial Agent
finance_agent = Agent(
    name='Finance AI Agent',
    model=Groq(id="llama-3.3-70b-versatile",api_key=GROQ_API_KEY),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[we_bsearch_agent, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile",api_key=GROQ_API_KEY),
    instructions=["Always include the sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarise analyst recommendations and share the latest news for NVIDA",stream=True)