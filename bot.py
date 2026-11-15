import os
import asyncio
import logging
import ccxt
import pandas as pd
import pandas_ta as ta
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI

# --- НАСТРОЙКИ ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
exchange = ccxt.binance()

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_news(symbol):
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_KEY}&currencies={symbol}&kind=news"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return [post['title'] for post in data.get('results', [])[:3]]
    except Exception:
        return ["Новости временно недоступны"]

def get_market_data(symbol, timeframe):
    try:
        ticker = symbol.upper() + "/USDT" if "USDT" not in symbol.upper() else symbol.upper()
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # Поиск Swing High/Low (локальные экстремумы)
        df['Swing_High'] = df['high'] == df['high'].rolling(11, center=True).max()
        df['Swing_Low'] = df['low'] == df['low'].rolling(11, center=True).min()
        
        last_swing_high = df[df['Swing_High']].iloc[-1]['high'] if not df[df['Swing_High']].empty else None
        last_swing_low = df[df['Swing_Low']].iloc[-1]['low'] if not df[df['Swing_Low']].empty else None
        
        return {
            "ticker": ticker,
            "current_price": df.iloc[-1]['close'],
            "rsi": round(df.iloc[-1]['RSI'], 2),
            "last_swing_high": last_swing_high,
            "last_swing_low": last_swing_low
        }
    except Exception as e:
        return {"error": str(e)}

# --- ОБРАБОТЧИКИ КОМАНД ---
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Я SMC-бот. Напиши мне тикер и таймфрейм, например: BTC 1h")

@dp.message()
async def analyze(message: types.Message):
    text = message.text.strip().split()
    if len(text) < 2:
        await message.answer("Напиши в формате: BTC 1h (тикер и таймфрейм)")
        return
    
    symbol, timeframe = text[0], text[1]
    await message.answer(f"Анализирую {symbol.upper()} на {timeframe}... (5-10 сек)")
    
    data = get_market_data(symbol, timeframe)
    if "error" in data:
        await message.answer(f"Ошибка: {data['error']}. Проверь тикер.")
        return
        
    news = get_news(symbol.upper())
    
    prompt = f"""
    Ты — профессиональный трейдер по SMC/ICT. Твоя задача — сделать анализ на основе данных.
    Данные: Текущая цена {data['current_price']}, RSI {data['rsi']}, Последний Swing High {data['last_swing_high']}, Последний Swing Low {data['last_swing_low']}.
    Новости: {', '.join(news) if news else 'Нет новостей'}.
    
    Напиши сценарий строго по правилам SMC.
    1. Определи, где находится цена относительно Swing High/Low.
    2. Опиши, куда ожидаем движение (лонг/шорт) и почему.
    3. Где ставить стоп-лосс и тейк-профит (соотношение риск/прибыль не менее 1:2).
    4. Укажи, какие новости могут отменить сценарий.
    Пиши кратко, профессионально, без воды. Используй термины Order Block, FVG, Liquidity.
    """
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        analysis_text = response.choices[0].message.content
        await message.answer(analysis_text)
    except Exception as e:
        await message.answer(f"Ошибка ИИ: {str(e)}")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
