import os
import asyncio
import logging
import io
import ccxt
import pandas as pd
import requests
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI

# --- НАСТРОЙКИ ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
exchange = ccxt.okx()  # OKX работает с серверами Railway без блокировок

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ==========================================
# SMC МАТЕМАТИКА
# ==========================================

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def find_swings(df, window=5):
    df['Swing_High'] = df['high'] == df['high'].rolling(window=window*2+1, center=True).max()
    df['Swing_Low'] = df['low'] == df['low'].rolling(window=window*2+1, center=True).min()
    return df

def find_order_blocks(df):
    obs = []
    avg_body = (df['close'] - df['open']).abs().rolling(20).mean()
    for i in range(2, len(df) - 1):
        body_next = abs(df['close'].iloc[i+1] - df['open'].iloc[i+1])
        avg = avg_body.iloc[i] if not pd.isna(avg_body.iloc[i]) else 1
        if df['close'].iloc[i] < df['open'].iloc[i]:
            if df['close'].iloc[i+1] > df['open'].iloc[i+1] and body_next > avg * 1.5:
                obs.append({'type': 'bullish', 'top': df['high'].iloc[i], 'bottom': df['low'].iloc[i], 'index': i})
        elif df['close'].iloc[i] > df['open'].iloc[i]:
            if df['close'].iloc[i+1] < df['open'].iloc[i+1] and body_next > avg * 1.5:
                obs.append({'type': 'bearish', 'top': df['high'].iloc[i], 'bottom': df['low'].iloc[i], 'index': i})
    return obs[-5:] if len(obs) > 5 else obs

def find_fvgs(df):
    fvgs = []
    for i in range(1, len(df) - 1):
        if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
            fvgs.append({'type': 'bullish', 'top': df['low'].iloc[i+1], 'bottom': df['high'].iloc[i-1], 'index': i})
        elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
            fvgs.append({'type': 'bearish', 'top': df['low'].iloc[i-1], 'bottom': df['high'].iloc[i+1], 'index': i})
    return fvgs[-5:] if len(fvgs) > 5 else fvgs

def find_liquidity_sweeps(df, lookback=10):
    sweeps = []
    for i in range(lookback, len(df) - 1):
        prev_high = df['high'].iloc[i-lookback:i].max()
        prev_low = df['low'].iloc[i-lookback:i].min()
        if df['high'].iloc[i] > prev_high and df['close'].iloc[i] < prev_high:
            sweeps.append({'type': 'bearish_sweep', 'price': df['high'].iloc[i], 'index': i})
        if df['low'].iloc[i] < prev_low and df['close'].iloc[i] > prev_low:
            sweeps.append({'type': 'bullish_sweep', 'price': df['low'].iloc[i], 'index': i})
    return sweeps[-3:] if len(sweeps) > 3 else sweeps

def get_premium_discount(df):
    swings_high = df[df['Swing_High']]['high']
    swings_low = df[df['Swing_Low']]['low']
    if len(swings_high) == 0 or len(swings_low) == 0:
        return None
    last_high = swings_high.iloc[-1]
    last_low = swings_low.iloc[-1]
    eq = (last_high + last_low) / 2
    current = df['close'].iloc[-1]
    range_size = last_high - last_low
    return {
        'high': last_high, 'low': last_low, 'equilibrium': eq,
        'zone': 'Premium' if current > eq else 'Discount',
        'ote_top': last_low + range_size * 0.79, 'ote_bottom': last_low + range_size * 0.62,
        'current': current
    }

# ==========================================
# ОТРИСОВКА ГРАФИКА
# ==========================================

def draw_chart(df, ticker, obs, fvgs, pd_data, sweeps):
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot['time'], unit='ms')
    df_plot = df_plot[['open', 'high', 'low', 'close', 'volume']]
    df_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=True, facecolor='#1e222d', figcolor='#1e222d')
    
    fig, axes = mpf.plot(df_plot, type='candle', style=s, volume=True, returnfig=True, figsize=(14, 8), tight_layout=True)
    ax = axes[0]
    
    if pd_data:
        ax.axhspan(pd_data['equilibrium'], pd_data['high'], alpha=0.15, color='red')
        ax.axhspan(pd_data['low'], pd_data['equilibrium'], alpha=0.15, color='green')
        ax.axhline(y=pd_data['equilibrium'], color='yellow', linestyle='--', linewidth=1)
        ax.axhspan(pd_data['ote_bottom'], pd_data['ote_top'], alpha=0.2, color='orange')
    
    for ob in obs:
        color = 'green' if ob['type'] == 'bullish' else 'red'
        ax.add_patch(Rectangle((ob['index'], ob['bottom']), 10, ob['top'] - ob['bottom'], linewidth=1, edgecolor=color, facecolor=color, alpha=0.3))
    
    for fvg in fvgs:
        color = 'lime' if fvg['type'] == 'bullish' else 'salmon'
        ax.add_patch(Rectangle((fvg['index'], fvg['bottom']), 8, fvg['top'] - fvg['bottom'], linewidth=1, edgecolor=color, facecolor=color, alpha=0.2, linestyle='--'))
    
    for sw in sweeps:
        marker = 'v' if 'bearish' in sw['type'] else '^'
        color = 'red' if 'bearish' in sw['type'] else 'green'
        ax.scatter(sw['index'], sw['price'], marker=marker, color=color, s=100, zorder=5)
    
    ax.set_title(f'{ticker} — SMC Analysis', color='white', fontsize=14)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1e222d')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==========================================
# ОСНОВНАЯ ЛОГИКА
# ==========================================

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("SMC Bot готов. Напиши: BTC 1h или ETH 4h")

@dp.message()
async def analyze(message: types.Message):
    text = message.text.strip().split()
    if len(text) < 2:
        await message.answer("Формат: BTC 1h")
        return
    
    raw_symbol, timeframe = text[0].upper(), text[1]
    
    # Очищаем символ от USDT, если пользователь написал BTCUSDT
    clean_symbol = raw_symbol.replace("USDT", "")
    
    # Приводим таймфрейм к нижнему регистру, как требует ccxt (4H -> 4h)
    okx_timeframe = timeframe.lower()
    
    await message.answer(f"Считаю SMC для {clean_symbol} на {okx_timeframe}...")
    
    try:
        # Формируем правильный тикер для OKX (BTC/USDT)
        ticker = f"{clean_symbol}/USDT"
        ohlcv = exchange.fetch_ohlcv(ticker, okx_timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        df['RSI'] = calculate_rsi(df['close'])
        current_rsi = round(df['RSI'].iloc[-1], 2)
        
        df = find_swings(df)
        obs = find_order_blocks(df)
        fvgs = find_fvgs(df)
        sweeps = find_liquidity_sweeps(df)
        pd_data = get_premium_discount(df)
        
        chart_buf = draw_chart(df, ticker, obs, fvgs, pd_data, sweeps)
        
        # --- БЕЗОПАСНОЕ ФОРМАТИРОВАНИЕ ---
        if pd_data:
            eq_str = f"{pd_data['equilibrium']:.2f}"
            ote_top_str = f"{pd_data['ote_top']:.2f}"
            ote_bot_str = f"{pd_data['ote_bottom']:.2f}"
            zone_str = pd_data['zone']
        else:
            eq_str = "N/A"
            ote_top_str = "N/A"
            ote_bot_str = "N/A"
            zone_str = "N/A"

        prompt = f"""
Ты — профессиональный ICT/SMC трейдер. Опиши сценарий на основе ГОТОВЫХ данных. НЕ ИЩИ УРОВНИ САМ.

Цена: {df['close'].iloc[-1]:.2f}
RSI(14): {current_rsi}
Зона: {zone_str}
Equilibrium (0.5): {eq_str}
OTE: {ote_bot_str} - {ote_top_str}

Order Blocks: {obs}
FVG: {fvgs}
Liquidity Sweeps: {sweeps}

ПРАВИЛА:
1. Если цена в Premium — только шорт. Если в Discount — только лонг.
2. Вход на пересечении OB и FVG.
3. Стоп за Swing High/Low. RR минимум 1:2.
4. Напиши Основной и Альтернативный сценарий.

Пиши кратко, профессионально, на русском.
"""
        
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        analysis = response.choices[0].message.content
        
        await message.answer_photo(photo=types.BufferedInputFile(chart_buf.getvalue(), filename="chart.png"))
        await message.answer(analysis)
        
    except Exception as e:
        await message.answer(f"Ошибка: {str(e)}")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
