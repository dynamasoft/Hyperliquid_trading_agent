from __future__ import annotations as _annotations
from newsapi import NewsApiClient
from utils import hyperliquid_util
import asyncio
import os
from dataclasses import dataclass
from typing import Any
import logfire
from devtools import debug
from httpx import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext
from dotenv import load_dotenv
from datetime import datetime, timedelta
from openai import OpenAI
from hyperliquid.utils import constants

load_dotenv()
LOG_FIRE_KEY = os.getenv("LOG_FIRE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logfire.configure(send_to_logfire=LOG_FIRE_KEY)
client = OpenAI(api_key=os.getenv(OPENAI_API_KEY))

# Initialize hyperliquid
address, info, exchange = hyperliquid_util.setup(
    base_url=constants.TESTNET_API_URL, skip_ws=True
)


@dataclass
class Deps:
    client: AsyncClient
    news_api_key: str | None


trade_agent = Agent(
    "openai:gpt-4o",
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    system_prompt=(
        "Be concise, reply with one sentence."
        "Use the `fetch_crypto_news` tool to get the news sentiment"
    ),
    deps_type=Deps,
    retries=2,
    instrument=True,
)


@trade_agent.tool
async def fetch_crypto_news(ctx: RunContext[Deps], crypto_currency: str) -> dict[str]:
    """Get the news about cryto currency.

    Args:
        crypto_currency
    """
    if ctx.deps.news_api_key is None:
        # if no API key is provided, return a dummy response (positive sentiment)
        return {"sentiment": "positive"}

    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    newsapi = NewsApiClient(api_key=ctx.deps.news_api_key)

    params = {
        "crypto_currency": crypto_currency,
        "from_param": from_date,
        "to": to_date,
        "language": "en",
        "sort_by": "relevancy",
        "page_size": 1,
    }

    with logfire.span("calling NewsApiClient API", params=params) as span:

        result = newsapi.get_everything(
            q=crypto_currency,
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="relevancy",
            page_size=1,
        )
        span.set_attribute("response", result)

    if result:
        return result["articles"]
    else:
        raise ModelRetry("Could not find news")


@trade_agent.tool
async def analyze_sentiment(ctx: RunContext[Deps], news_text: str):
    """Send news text to GPT-3.5 Turbo for sentiment analysis."""
    prompt = f"""
    Determine if the sentiment of the following news is Positive or Negative. Only respond with "Positive" or "Negative" and nothing else.
    
    News: "{news_text}"
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial sentiment analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )

    sentiment = response.choices[0].message.content.strip()
    return sentiment


@trade_agent.tool
def make_trade_decision(
    ctx: RunContext[Deps], overall_sentiment, asset, position_size, stop_loss_price
):
    """Use GPT-3.5 Turbo to decide trade action and return structured JSON output."""
    prompt = f"""
    Based on the sentiment analysis, determine whether to go LONG or SHORT on {asset}. 
    The position size is {position_size:.4f} {asset} and the stop loss price is {stop_loss_price:.2f}. 
    Respond strictly in JSON format with the following structure:
    {{
        "action": "Long" or "Short",
        "asset": "{asset}",
        "position_size": {position_size:.4f},
        "stop_loss_price": {stop_loss_price:.2f}
    }}
    
    Sentiment: {overall_sentiment}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial trading assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )

    trade_decision = response.choices[0].message.content.strip()
    try:
        return json.loads(trade_decision)
    except json.JSONDecodeError:
        return None


@trade_agent.tool
def trade(ctx: RunContext[Deps], decision: str):
    result = exchange.order("ETH", True, 0.0002, 1100, {"limit": {"tif": "Gtc"}})
    pass


async def main():
    async with AsyncClient() as client:

        news_api_key = os.getenv("NEWS_API_KEY")

        deps = Deps(client=client, news_api_key=news_api_key)

        result = await trade_agent.run(
            "Based on the crypto news about ETH, execute BUY, SELL or HOLD in hyperliquid platform?",
            deps=deps,
        )
        debug(result)
        print("Response:", result.data)


if __name__ == "__main__":
    asyncio.run(main())
