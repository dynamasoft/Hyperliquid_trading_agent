import json
from newsapi import NewsApiClient
from openai import OpenAI
from utils import hyperliquid_util
import os
from hyperliquid.utils import constants
from datetime import datetime, timedelta

# Set API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Set this in your environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set this in your environment variables
client = OpenAI(api_key=os.getenv(OPENAI_API_KEY))


def fetch_crypto_news(input: str) -> list:
    """Fetch latest crypto-related news from NewsAPI."""

    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # Auto-generate date range (last 7 days)
    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    # /v2/everything
    result = newsapi.get_everything(
        q=input,
        # sources="crypto-coins-news",
        # domains='bbc.co.uk,techcrunch.com',
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="relevancy",
        page_size=1,
    )

    return result["articles"]

    # url = f"https://newsapi.org/v2/everything?q=cryptocurrency&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    # response = requests.get(url)
    # data = response.json()

    # if "articles" in data:
    #     return [(article["title"], article["description"]) for article in data["articles"][:5]]  # Fetch top 5 articles
    # return []


def analyze_sentiment(news_text):
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


def aggregate_sentiments(sentiments):
    """Aggregate sentiment results to determine overall sentiment."""
    positive_count = sentiments.count("Positive")
    negative_count = sentiments.count("Negative")

    if positive_count > negative_count:
        return "Overall Positive"
    elif negative_count > positive_count:
        return "Overall Negative"
    else:
        return "Neutral (Equal Positive and Negative)"


def trade(decision: str):

    address, info, exchange = hyperliquid_util.setup(
        base_url=constants.TESTNET_API_URL, skip_ws=True
    )

    user_state = info.user_state(address)

    # cloid = Cloid.from_str("0x00000000000000000000000000000001")

    # Tif = Union[Literal["Alo"], Literal["Ioc"], Literal["Gtc"]]
    # result = exchange.order(
    #     name=decision["asset"].upper(),
    #     is_buy=False,  # Buy back to close short position
    #     sz=decision["position_size"],
    #     limit_px=decision["stop_loss_price"],
    #     order_type=TypedDict("LimitOrderType", {"tif": Tif}),
    # )

    result = exchange.order("ETH", True, 0.0002, 1100, {"limit": {"tif": "Gtc"}})

    # order_type=OrderType(
    #     {"stopLimit": {"stopPx": decision["stop_loss_price"], "tif": "Gtc"}}
    # )

    user_state1 = info.user_state(address)
    test = "ddd"

    # Get the user state and print out position information
    # user_state = info.user_state(address)
    # positions = []
    # for position in user_state["assetPositions"]:
    #     positions.append(position["position"])
    # if len(positions) > 0:
    #     print("positions:")
    #     for position in positions:
    #         print(json.dumps(position, indent=2))
    # else:
    #     print("no open positions")


def make_trade_decision(overall_sentiment, asset, position_size, stop_loss_price):
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


def fetch_account_balance():
    """Fetch account balance and positions from Hyperliquid."""
    address, info, exchange = hyperliquid_util.setup(
        base_url=constants.TESTNET_API_URL, skip_ws=True
    )
    user_state = info.user_state(address)
    balance = user_state["marginSummary"]["accountValue"]
    return float(balance)


def fetch_market_data(asset):
    """Fetch latest market price for the given asset from Hyperliquid."""
    # address, info, exchange = hyperliquid_util.setup(
    #     base_url=constants.TESTNET_API_URL, skip_ws=True
    # )
    # market_data = info.asset_price(asset)
    # return market_data.get("price", None)
    return 100


def calculate_position_size(balance, entry_price, stop_loss_percent=5, risk_percent=2):
    """Calculate position size (X) and stop loss level (Y)."""

    risk_amount = balance * (risk_percent / 100)  # Risk per trade
    stop_loss_price = entry_price * (1 - stop_loss_percent / 100)  # Stop loss level
    distance_to_stop_loss = entry_price - stop_loss_price
    position_size = (
        risk_amount / distance_to_stop_loss
    )  # X = Risk Amount / Distance to Stop Loss
    return position_size, stop_loss_price


def main():

    search = "eth"
    news_articles = fetch_crypto_news(search)
    sentiments = []
    for article in news_articles:
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")
        sentiment = analyze_sentiment(description)
        sentiments.append(sentiment)

    overall_sentiment = aggregate_sentiments(sentiments)

    # print(f"overall Sentiment: {overall_sentiment}\n")

    balance = fetch_account_balance()
    entry_price = fetch_market_data(search.upper())
    position_size, stop_loss_price = calculate_position_size(balance, entry_price)
    trade_decision = make_trade_decision(
        overall_sentiment, search.upper(), position_size, stop_loss_price
    )

    trade(trade_decision)
    test = "33"
    test2 = test


if __name__ == "__main__":
    main()
