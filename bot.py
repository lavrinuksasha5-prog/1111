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
exchange = ccxt.okx()  # OKX стабильно работает с Railway

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
# ОТРИСОВКА ГРАФИКА (ВИЗУАЛ TRADINGVIEW + СТРЕЛКИ)
# ==========================================

def draw_chart(df, ticker, obs, fvgs, pd_data, sweeps):
    df_plot = df.tail(80).copy()
    df_plot.index = pd.to_datetime(df_plot['time'], unit='ms')
    df_plot = df_plot[['open', 'high', 'low', 'close', 'volume']]
    df_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Черно-белые свечи, как на TradingView
    mc = mpf.make_marketcolors(up='white', down='black', edge='black', wick='black', volume='gray')
    s = mpf.make_mpf_style(
        marketcolors=mc, gridstyle='', y_on_right=True, facecolor='white', figcolor='white',
        rc={'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'}
    )
    
    fig, axes = mpf.plot(df_plot, type='candle', style=s, volume=True, 
                         returnfig=True, figsize=(16, 9), tight_layout=True,
                         scale_padding={'left': 0.5, 'right': 3.0, 'top': 0.5, 'bottom': 0.5})
    ax = axes[0]
    axes[2].set_visible(False)
    
    offset = len(df) - len(df_plot)
    current_price = df['close'].iloc[-1]
    
    # Зоны Premium / Discount
    if pd_data:
        ax.axhspan(pd_data['equilibrium'], pd_data['high'], alpha=0.1, color='gray', zorder=0)
        ax.axhspan(pd_data['low'], pd_data['equilibrium'], alpha=0.25, color='gray', zorder=0)
        ax.axhline(y=pd_data['equilibrium'], color='black', linestyle='-', linewidth=1, zorder=1)
        ax.text(len(df_plot) + 1, pd_data['equilibrium'], '0.5', va='center', ha='left', fontsize=10, color='black')
        
    # Order Blocks
    for ob in obs:
        if ob['index'] >= offset:
            ax.add_patch(Rectangle((ob['index'] - offset, ob['bottom']), 15, ob['top'] - ob['bottom'], 
                                   linewidth=0, facecolor='gray', alpha=0.3, zorder=0))
            
    # --- ЛОГИКА ДЛЯ СТРЕЛОК ---
    is_long = True if pd_data and pd_data['zone'] == 'Discount' else False
    target_ob = None
    
    if is_long:
        valid_obs = [ob for ob in obs if ob['type'] == 'bullish' and ob['top'] < current_price]
        if valid_obs: target_ob = valid_obs[-1]
    else:
        valid_obs = [ob for ob in obs if ob['type'] == 'bearish' and ob['bottom'] > current_price]
        if valid_obs: target_ob = valid_obs[-1]
            
    x_start = len(df_plot) - 1
    
    if target_ob:
        entry_price = (target_ob['top'] + target_ob['bottom']) / 2
        
        if is_long:
            take_profit = pd_data['equilibrium'] if pd_data else current_price * 1.02
            x_vals = [x_start, x_start + 4, x_start + 8, x_start + 12]
            y_vals = [current_price, entry_price, entry_price, take_profit]
            
            ax.plot([x_start + 4, x_start + 12], [target_ob['bottom'], target_ob['bottom']], color='red', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.text(x_start + 13, target_ob['bottom'], 'SL', color='red', fontsize=10, va='center')
            
        else:
            take_profit = pd_data['equilibrium'] if pd_data else current_price * 0.98
            x_vals = [x_start, x_start + 4, x_start + 8, x_start + 12]
            y_vals = [current_price, entry_price, entry_price, take_profit]
            
            ax.plot([x_start + 4, x_start + 12], [target_ob['top'], target_ob['top']], color='red', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.text(x_start + 13, target_ob['top'], 'SL', color='red', fontsize=10, va='center')
            
        ax.plot(x_vals, y_vals, color='black', linewidth=2, zorder=5)
        ax.annotate('', xy=(x_vals[-1], y_vals[-1]), xytext=(x_vals[-2], y_vals[-2]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(x_start + 13, take_profit, 'TP', color='green', fontsize=10, va='center')
        
    else:
        eq = pd_data['equilibrium'] if pd_data else current_price * 1.01
        x_vals = [x_start, x_start + 4, x_start + 8]
        y_vals = [current_price, eq, eq * 0.99]
        ax.plot(x_vals, y_vals, color='black', linewidth=2, zorder=5, linestyle=':')
        ax.annotate('', xy=(x_vals[-1], y_vals[-1]), xytext=(x_vals[-2], y_vals[-2]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Метки текущей цены
    ax.text(len(df_plot) + 1, current_price, f'{current_price:.1f}', 
            va='center', ha='left', fontsize=10, color='white', 
            bbox=dict(facecolor='gray', edgecolor='gray', boxstyle='square,pad=0.2'))
    
    # Оформление
    ax.set_title('', color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(color='#f0f0f0', linestyle='-', linewidth=0.5)

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
        
        if pd_data:
            eq_str = f"{pd_data['equilibrium']:.2f}"
            ote_top_str = f"{pd_data['ote_top']:.2f}"
            ote_bot_str = f"{pd_data['ote_bottom']:.2f}"
            zone_str = pd_data['zone']
        else:
            eq_str = ote_top_str = ote_bot_str = zone_str = "N/A"

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
