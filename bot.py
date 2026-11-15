import os
import asyncio
import logging
import io
import re
import json
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
exchange = ccxt.okx()  # OKX стабильно работает с Railway

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ==========================================
# SMC МАТЕМАТИКА (ЯДРО)
# ==========================================

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_htf_bias(symbol):
    """LTP: Определяет глобальный тренд на 4H"""
    try:
        ohlcv = exchange.fetch_ohlcv(f"{symbol}/USDT", '4h', limit=50)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['Swing_High'] = df['high'] == df['high'].rolling(window=5, center=True).max()
        df['Swing_Low'] = df['low'] == df['low'].rolling(window=5, center=True).min()
        
        highs = df[df['Swing_High']]['high'].tolist()
        lows = df[df['Swing_Low']]['low'].tolist()
        
        if len(highs) >= 3 and len(lows) >= 3:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                return "BULLISH (HH/HL)"
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                return "BEARISH (LH/LL)"
        return "RANGING"
    except:
        return "UNKNOWN"

def find_market_structure(df, window=3):
    """ITP: Определяет локальную структуру рынка (BOS/CHoCH)"""
    df['Swing_High'] = df['high'] == df['high'].rolling(window=window*2+1, center=True).max()
    df['Swing_Low'] = df['low'] == df['low'].rolling(window=window*2+1, center=True).min()
    
    highs = df[df['Swing_High']]['high'].tolist()
    lows = df[df['Swing_Low']]['low'].tolist()
    
    structure = "UNKNOWN"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            structure = "BULLISH"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            structure = "BEARISH"
            
    return df, structure, highs[-1] if highs else None, lows[-1] if lows else None

def find_order_blocks(df):
    """POI: Ищет Order Blocks (последняя противоположная свеча перед импульсом)"""
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
    """POI: Ищет Fair Value Gaps (имбалансы)"""
    fvgs = []
    for i in range(1, len(df) - 1):
        if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
            fvgs.append({'type': 'bullish', 'top': df['low'].iloc[i+1], 'bottom': df['high'].iloc[i-1], 'index': i})
        elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
            fvgs.append({'type': 'bearish', 'top': df['low'].iloc[i-1], 'bottom': df['high'].iloc[i+1], 'index': i})
    return fvgs[-5:] if len(fvgs) > 5 else fvgs

def find_liquidity_sweeps(df, lookback=5):
    """Manipulation: Ищет свипы ликвидности (Raid/Sweep)"""
    sweeps = []
    for i in range(lookback, len(df) - 1):
        prev_high = df['high'].iloc[i-lookback:i].max()
        if df['high'].iloc[i] > prev_high and df['close'].iloc[i] < prev_high:
            sweeps.append({'type': 'bearish_sweep', 'price': df['high'].iloc[i], 'index': i})
        
        prev_low = df['low'].iloc[i-lookback:i].min()
        if df['low'].iloc[i] < prev_low and df['close'].iloc[i] > prev_low:
            sweeps.append({'type': 'bullish_sweep', 'price': df['low'].iloc[i], 'index': i})
    return sweeps[-3:] if len(sweeps) > 3 else sweeps

def get_premium_discount(df):
    """PD Array: Premium / Discount"""
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
# ОТРИСОВКА ГРАФИКА (ВИЗУАЛ TRADINGVIEW)
# ==========================================

def draw_chart(df, ticker, obs, fvgs, pd_data, sweeps, trade_data):
    df_plot = df.tail(70).copy()
    df_plot.index = pd.to_datetime(df_plot['time'], unit='ms')
    df_plot = df_plot[['open', 'high', 'low', 'close', 'volume']]
    df_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    mc = mpf.make_marketcolors(up='white', down='#0a192f', edge='#0a192f', wick='#0a192f', volume='gray')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=True, facecolor='white', figcolor='white',
                           rc={'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})
    
    fig, axes = mpf.plot(df_plot, type='candle', style=s, volume=False, 
                         returnfig=True, figsize=(16, 9), tight_layout=True,
                         scale_padding={'left': 0.5, 'right': 3.5, 'top': 0.5, 'bottom': 0.5})
    ax = axes[0]
    
    offset = len(df) - len(df_plot)
    current_price = df['close'].iloc[-1]
    
    # Голубые зоны имбаланса
    for fvg in fvgs:
        if fvg['index'] >= offset:
            ax.add_patch(Rectangle((fvg['index'] - offset, fvg['bottom']), 30, fvg['top'] - fvg['bottom'], 
                                   linewidth=0, facecolor='#88d8e8', alpha=0.5, zorder=0))
            
    # Черные линии уровней
    if pd_data:
        ax.axhline(y=pd_data['high'], color='black', linestyle='-', linewidth=1.5, zorder=1)
        ax.axhline(y=pd_data['low'], color='black', linestyle='-', linewidth=1.5, zorder=1)
        
        ax.text(len(df_plot) + 1, pd_data['high'], f"{pd_data['high']:.1f}", va='center', ha='left', fontsize=10, color='white', bbox=dict(facecolor='black', edgecolor='black', boxstyle='square,pad=0.2'))
        ax.text(len(df_plot) + 1, pd_data['low'], f"{pd_data['low']:.1f}", va='center', ha='left', fontsize=10, color='white', bbox=dict(facecolor='black', edgecolor='black', boxstyle='square,pad=0.2'))

    # Сценарий (Стрелки)
    if trade_data:
        entry = float(trade_data.get('entry', current_price))
        sl = float(trade_data.get('sl', current_price * 0.98))
        tp = float(trade_data.get('tp', current_price * 1.02))
        direction = trade_data.get('direction', 'SHORT').upper()
        
        x_start = len(df_plot) - 1
        
        if direction == 'SHORT':
            x_vals = [x_start, x_start + 4, x_start + 9, x_start + 14]
            y_vals = [current_price, entry, entry * 0.998, tp] 
        else:
            x_vals = [x_start, x_start + 4, x_start + 9, x_start + 14]
            y_vals = [current_price, entry, entry * 1.002, tp]

        ax.plot(x_vals, y_vals, color='black', linewidth=2, zorder=5)
        ax.annotate('', xy=(x_vals[-1], y_vals[-1]), xytext=(x_vals[-2], y_vals[-2]), arrowprops=dict(arrowstyle='->', color='black', lw=2))

        ax.axhline(y=sl, color='red', linestyle='-', linewidth=1.5, alpha=0.8, zorder=2)
        ax.text(len(df_plot) + 1, sl, f"SL {sl:.1f}", color='white', fontsize=10, va='center', bbox=dict(facecolor='red', edgecolor='red', boxstyle='square,pad=0.2'))
        
        ax.axhline(y=tp, color='green', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
        ax.text(len(df_plot) + 1, tp, f"TP {tp:.1f}", color='white', fontsize=10, va='center', bbox=dict(facecolor='green', edgecolor='green', boxstyle='square,pad=0.2'))

    # Текущая цена
    ax.text(len(df_plot) + 1, current_price, f'{current_price:.1f}', 
            va='center', ha='left', fontsize=10, color='white', 
            bbox=dict(facecolor='gray', edgecolor='gray', boxstyle='square,pad=0.2'))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
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
    clean_symbol = raw_symbol.replace("USDT", "")
    okx_timeframe = timeframe.lower()
    
    await message.answer(f"Считаю SMC для {clean_symbol} на {okx_timeframe}...")
    
    try:
        ticker = f"{clean_symbol}/USDT"
        
        # 1. LTP (Контекст старшего ТФ)
        htf_bias = get_htf_bias(clean_symbol)
        
        # 2. Данные для младшего ТФ
        ohlcv = exchange.fetch_ohlcv(ticker, okx_timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        df['RSI'] = calculate_rsi(df['close'])
        current_rsi = round(df['RSI'].iloc[-1], 2)
        
        df, structure, last_high, last_low = find_market_structure(df)
        obs = find_order_blocks(df)
        fvgs = find_fvgs(df)
        sweeps = find_liquidity_sweeps(df)
        pd_data = get_premium_discount(df)
        
        eq_str = f"{pd_data['equilibrium']:.2f}" if pd_data else "N/A"
        zone_str = pd_data['zone'] if pd_data else "N/A"

        # 3. Промпт по методичке (Top-Down + PO3)
        prompt = f"""
Ты — профессиональный трейдер по Smart Money Concept (SMC). Твоя задача — не гадать, а читать поток ордеров (Order Flow) по методичке.

ИЕРАРХИЯ АНАЛИЗА (Top-Down):
1. LTP (Контекст старшего ТФ 4H): {htf_bias}
2. ITP (Структура на текущем ТФ): {structure}
3. Ликвидность: Свипы (Raid/Sweep): {sweeps}
4. POI (Точки интереса): Order Blocks: {obs}, FVG: {fvgs}
5. Зона: {zone_str} (EQ: {eq_str})

ПРАВИЛА ПОСТРОЕНИЯ СДЕЛКИ:
- Если HTF BIAS = BEARISH, мы ищем ТОЛЬКО SHORT. Если BULLISH — ТОЛЬКО LONG. (Против тренда не торгуем).
- Сделка должна быть от Манипуляции к Распределению (PO3). Обязательно должен быть недавний свип (Sweep) ликвидности.
- ВХОД: Строго от 50% (Mean Threshold) или края ближайшего FVG / OB, который остался после свипа.
- СТОП: Строго за тень свип-хая (для шорта) или свип-лоя (для лонга). Стоп в 5 баксов от входа — это ошибка!
- ТЕЙК: На противоположной ликвидности или полном заполнении (Full Fill) противоположного FVG.

ФОРМАТ ОТВЕТА:
Сначала строго JSON (без ```json):
{{"direction": "SHORT", "entry": 80200.0, "sl": 81000.0, "tp": 78000.0}}

Затем с новой строки напиши анализ по-русски:
**Narrative (Сюжет):** Кратко опиши, что делает крупный игрок (накопление, манипуляция, распределение).
**Ключевая зона — [Цена]$.**
Объясни, где был свип, почему выбран этот FVG для входа и почему HTF BIAS подтверждает сделку.
"""
        
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 
        )
        raw_response = response.choices[0].message.content
        
        # Парсинг JSON
        trade_data = None
        analysis_text = raw_response
        try:
            json_match = re.search(r'\{.*?\}', raw_response)
            if json_match:
                trade_data = json.loads(json_match.group(0))
                analysis_text = raw_response.replace(json_match.group(0), "").strip()
        except: pass
        
        chart_buf = draw_chart(df, ticker, obs, fvgs, pd_data, sweeps, trade_data)
        
        await message.answer_photo(photo=types.BufferedInputFile(chart_buf.getvalue(), filename="chart.png"))
        await message.answer(analysis_text)
        
    except Exception as e:
        await message.answer(f"Ошибка: {str(e)}")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
