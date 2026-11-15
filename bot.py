import os
import asyncio
import logging
import io
import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI

# --- НАСТРОЙКИ (БЕРУТСЯ ИЗ RAILWAY VARIABLES) ---
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

# ==========================================
# SMC МАТЕМАТИКА (ЯДРО)
# ==========================================

def find_swings(df, window=5):
    """Находит Swing High/Low"""
    df['Swing_High'] = df['high'] == df['high'].rolling(window=window*2+1, center=True).max()
    df['Swing_Low'] = df['low'] == df['low'].rolling(window=window*2+1, center=True).min()
    return df

def find_order_blocks(df):
    """Ищет Order Blocks (последняя противоположная свеча перед импульсом)"""
    obs = []
    avg_body = (df['close'] - df['open']).abs().rolling(20).mean()
    
    for i in range(2, len(df) - 1):
        body_current = abs(df['close'].iloc[i] - df['open'].iloc[i])
        body_next = abs(df['close'].iloc[i+1] - df['open'].iloc[i+1])
        avg = avg_body.iloc[i] if not pd.isna(avg_body.iloc[i]) else 1
        
        # Бычий OB: медвежья свеча + следующая бычья с импульсом
        if df['close'].iloc[i] < df['open'].iloc[i]:  # Медвежья
            if df['close'].iloc[i+1] > df['open'].iloc[i+1]:  # Следующая бычья
                if body_next > avg * 1.5:  # Импульс
                    obs.append({
                        'type': 'bullish', 'top': df['high'].iloc[i], 
                        'bottom': df['low'].iloc[i], 'index': i
                    })
        # Медвежий OB: бычья свеча + следующая медвежья с импульсом
        elif df['close'].iloc[i] > df['open'].iloc[i]:  # Бычья
            if df['close'].iloc[i+1] < df['open'].iloc[i+1]:  # Следующая медвежья
                if body_next > avg * 1.5:
                    obs.append({
                        'type': 'bearish', 'top': df['high'].iloc[i], 
                        'bottom': df['low'].iloc[i], 'index': i
                    })
    return obs[-5:] if len(obs) > 5 else obs

def find_fvgs(df):
    """Ищет Fair Value Gaps (имбалансы)"""
    fvgs = []
    for i in range(1, len(df) - 1):
        # Бычий FVG: low 3-й свечи > high 1-й
        if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
            fvgs.append({
                'type': 'bullish', 'top': df['low'].iloc[i+1], 
                'bottom': df['high'].iloc[i-1], 'index': i
            })
        # Медвежий FVG: high 3-й свечи < low 1-й
        elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
            fvgs.append({
                'type': 'bearish', 'top': df['low'].iloc[i-1], 
                'bottom': df['high'].iloc[i+1], 'index': i
            })
    return fvgs[-5:] if len(fvgs) > 5 else fvgs

def find_liquidity_sweeps(df, lookback=10):
    """Ищет Liquidity Sweeps (ложные пробои)"""
    sweeps = []
    for i in range(lookback, len(df) - 1):
        prev_high = df['high'].iloc[i-lookback:i].max()
        prev_low = df['low'].iloc[i-lookback:i].min()
        
        # Свип над хаем (медвежий)
        if df['high'].iloc[i] > prev_high and df['close'].iloc[i] < prev_high:
            sweeps.append({'type': 'bearish_sweep', 'price': df['high'].iloc[i], 'index': i})
        # Свип под лоем (бычий)
        if df['low'].iloc[i] < prev_low and df['close'].iloc[i] > prev_low:
            sweeps.append({'type': 'bullish_sweep', 'price': df['low'].iloc[i], 'index': i})
    return sweeps[-3:] if len(sweeps) > 3 else sweeps

def get_premium_discount(df):
    """Premium/Discount + OTE"""
    swings_high = df[df['Swing_High']]['high']
    swings_low = df[df['Swing_Low']]['low']
    
    if len(swings_high) == 0 or len(swings_low) == 0:
        return None
    
    last_high = swings_high.iloc[-1]
    last_low = swings_low.iloc[-1]
    eq = (last_high + last_low) / 2
    current = df['close'].iloc[-1]
    
    # OTE (Optimal Trade Entry) 62-79%
    range_size = last_high - last_low
    ote_top = last_low + range_size * 0.79
    ote_bottom = last_low + range_size * 0.62
    
    return {
        'high': last_high, 'low': last_low, 'equilibrium': eq,
        'zone': 'Premium' if current > eq else 'Discount',
        'ote_top': ote_top, 'ote_bottom': ote_bottom,
        'current': current
    }

# ==========================================
# ОТРИСОВКА ГРАФИКА
# ==========================================

def draw_chart(df, ticker, obs, fvgs, pd_data, sweeps):
    """Рисует свечи + зоны + уровни"""
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot['time'], unit='ms')
    df_plot = df_plot[['open', 'high', 'low', 'close', 'volume']]
    df_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Стиль
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=True, facecolor='#1e222d', figcolor='#1e222d')
    
    fig, axes = mpf.plot(df_plot, type='candle', style=s, volume=True, 
                         returnfig=True, figsize=(14, 8), tight_layout=True)
    ax = axes[0]
    
    # Рисуем Premium/Discount
    if pd_data:
        # Premium зона (верхняя половина)
        ax.axhspan(pd_data['equilibrium'], pd_data['high'], alpha=0.15, color='red', label='Premium')
        # Discount зона (нижняя половина)
        ax.axhspan(pd_data['low'], pd_data['equilibrium'], alpha=0.15, color='green', label='Discount')
        # Equilibrium
        ax.axhline(y=pd_data['equilibrium'], color='yellow', linestyle='--', linewidth=1, label='EQ 0.5')
        # OTE
        ax.axhspan(pd_data['ote_bottom'], pd_data['ote_top'], alpha=0.2, color='orange', label='OTE')
    
    # Рисуем Order Blocks
    for ob in obs:
        color = 'green' if ob['type'] == 'bullish' else 'red'
        rect = Rectangle((ob['index'], ob['bottom']), 10, ob['top'] - ob['bottom'],
                         linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(rect)
    
    # Рисуем FVG
    for fvg in fvgs:
        color = 'lime' if fvg['type'] == 'bullish' else 'salmon'
        rect = Rectangle((fvg['index'], fvg['bottom']), 8, fvg['top'] - fvg['bottom'],
                         linewidth=1, edgecolor=color, facecolor=color, alpha=0.2, linestyle='--')
        ax.add_patch(rect)
    
    # Рисуем Sweeps (метки)
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

def get_news(symbol):
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_KEY}&currencies={symbol}&kind=news"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return [post['title'] for post in data.get('results', [])[:3]]
    except Exception:
        return ["Новости недоступны"]

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("SMC Bot готов. Напиши: BTC 1h или ETH 4h")

@dp.message()
async def analyze(message: types.Message):
    text = message.text.strip().split()
    if len(text) < 2:
        await message.answer("Формат: BTC 1h")
        return
    
    symbol, timeframe = text[0].upper(), text[1]
    await message.answer(f"Считаю SMC для {symbol} на {timeframe}...")
    
    try:
        ticker = symbol + "/USDT" if "USDT" not in symbol else symbol
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # SMC математика
        df = find_swings(df)
        obs = find_order_blocks(df)
        fvgs = find_fvgs(df)
        sweeps = find_liquidity_sweeps(df)
        pd_data = get_premium_discount(df)
        
        # Новости
        news = get_news(symbol)
        
        # Рисуем график
        chart_buf = draw_chart(df, ticker, obs, fvgs, pd_data, sweeps)
        
        # Промпт для DeepSeek
        prompt = f"""
Ты — профессиональный ICT/SMC трейдер. Опиши сценарий на основе ГОТОВЫХ данных. НЕ ИЩИ УРОВНИ САМ.

Цена: {df['close'].iloc[-1]:.2f}
Зона: {pd_data['zone'] if pd_data else 'N/A'}
Equilibrium (0.5): {pd_data['equilibrium']:.2f if pd_data else 'N/A'}
OTE: {pd_data['ote_bottom']:.2f} - {pd_data['ote_top']:.2f if pd_data else 'N/A'}

Order Blocks: {obs}
FVG: {fvgs}
Liquidity Sweeps: {sweeps}
Новости: {', '.join(news)}

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
        
        # Отправляем картинку + текст
        await message.answer_photo(photo=types.BufferedInputFile(chart_buf.getvalue(), filename="chart.png"))
        await message.answer(analysis)
        
    except Exception as e:
        await message.answer(f"Ошибка: {str(e)}")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
