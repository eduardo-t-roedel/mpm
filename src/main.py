import time
import pandas as pd
from tabulate import tabulate
import re
import talib as ta
import numpy as np
import os

def get_cache_filename_for_candles(timeframe_str):
    return f"candles_{timeframe_str}.parquet"

def get_cache_filename_for_csv():
    return "original_data.parquet"

def read_csv_if_needed(csv_path):
    csv_cache_path = get_cache_filename_for_csv()
    if os.path.exists(csv_cache_path):
        print("Carregando dados originais do CSV em cache...")
        df = pd.read_parquet(csv_cache_path)
    else:
        print("Lendo CSV...")
        start_time = time.perf_counter()
        df = pd.read_csv(
            csv_path,
            usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
            parse_dates=["Date"],
            date_format="%Y-%m-%d %H:%M:%S%z",
            dtype={
                "Open": "float64",
                "High": "float64",
                "Low": "float64",
                "Close": "float64",
                "Volume": "float64"
            },
            memory_map=True
        )
        end_time = time.perf_counter()
        print(f"Tempo para ler o CSV: {end_time - start_time:.6f} segundos")
        df.to_parquet(csv_cache_path)
    return df

def read_and_cache_candles(df, timeframe_str):
    cache_path = get_cache_filename_for_candles(timeframe_str)
    if os.path.exists(cache_path):
        print(f"Carregando candles em cache para timeframe {timeframe_str}...")
        candles = pd.read_parquet(cache_path)
        return candles
    else:
        print(f"Criando candles para timeframe {timeframe_str}...")
        candles = create_candles(df, timeframe_str)
        candles.to_parquet(cache_path)
        return candles

def create_candles(df, timeframe_str):
    units_map = {
        'min': 'min',
        'h': 'h',
        'd': 'd',
        'w': 'w',
        'm': 'm',
        'y': 'y'
    }

    tf_str = timeframe_str.lower().strip()
    match = re.match(r"(\d+)([a-zA-Z]+)", tf_str)
    if not match:
        raise ValueError("Formato de timeframe inválido. Exemplo: '210min', '1H', etc.")
    value, unit = match.groups()
    unit = unit.lower()
    if unit not in units_map:
        raise ValueError(f"Unidade de timeframe não suportada: {unit}")

    pandas_tf = value + units_map[unit]

    df = df.set_index("Date").sort_index()
    candles = df.resample(pandas_tf).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna(how='any')
    return candles

def format_duration_minutes(duration_minutes):
    total_seconds = int(duration_minutes * 60)
    months = total_seconds // (30*24*3600)
    remainder = total_seconds % (30*24*3600)
    days = remainder // (24*3600)
    remainder %= (24*3600)
    hours = remainder // 3600
    remainder %= 3600
    minutes = remainder // 60
    seconds = remainder % 60
    return f"{months:02d}:{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"

def rma(data, period):
    return ta.EMA(data, timeperiod=period)

def zlema(data, period):
    lag = (period - 1) / 2
    data_shifted = data.shift(int(lag))
    data_adjusted = data + (data - data_shifted)
    return ta.EMA(data_adjusted.values, timeperiod=period)

def hma(data, period):
    half_period = int(period/2)
    sqrt_period = int(np.sqrt(period))
    wma_half = ta.WMA(data, timeperiod=half_period)
    wma_full = ta.WMA(data, timeperiod=period)
    hull = 2*wma_half - wma_full
    return ta.WMA(hull, timeperiod=sqrt_period)

def alma(data, period, offset=0.85, sigma=6):
    m = (offset * (period - 1))
    s = period/sigma
    wts = [np.exp(-((i - m)**2)/(2*s*s)) for i in range(period)]
    wts = wts/np.sum(wts)
    res = data.rolling(period).apply(lambda x: np.sum(x*wts), raw=True)
    return res

def wilders(data, period):
    alpha = 1/period
    return data.ewm(alpha=alpha, adjust=False).mean()

def linreg(data, period):
    def linreg_calc(x):
        y = x.values
        x_idx = np.arange(len(y))
        slope, intercept = np.polyfit(x_idx, y, 1)
        return intercept + slope*(len(y)-1)
    return data.rolling(period).apply(linreg_calc, raw=False)

def frama(data, period):
    return ta.KAMA(data, timeperiod=period)

def jma(data, period):
    return ta.EMA(data, timeperiod=period)

def swma(data, period):
    w = ta.WMA(data, timeperiod=period)
    return ta.EMA(w, timeperiod=period)

def median_ma(data, period):
    return data.rolling(period).median()

def create_ma_line(data, ma_type='SMA', period=14):
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    ma_type = ma_type.upper().strip()
    ma_functions = {
        'SMA':       lambda d,p: ta.SMA(d, timeperiod=p),
        'EMA':       lambda d,p: ta.EMA(d, timeperiod=p),
        'WMA':       lambda d,p: ta.WMA(d, timeperiod=p),
        'DEMA':      lambda d,p: ta.DEMA(d, timeperiod=p),
        'TEMA':      lambda d,p: ta.TEMA(d, timeperiod=p),
        'TRIMA':     lambda d,p: ta.TRIMA(d, timeperiod=p),
        'KAMA':      lambda d,p: ta.KAMA(d, timeperiod=p),
        'MAMA':      lambda d,p: ta.MAMA(d, fastlimit=0.5, slowlimit=0.05)[0],
        'T3':        lambda d,p: ta.T3(d, timeperiod=p),
        'HT_TRENDLINE': lambda d,p: ta.HT_TRENDLINE(d),
        'RMA':       lambda d,p: rma(d,p),
        'ZLEMA':     lambda d,p: zlema(d,p),
        'HMA':       lambda d,p: hma(d,p),
        'ALMA':      lambda d,p: alma(d,p),
        'WILDERS':   lambda d,p: wilders(d,p),
        'LINREG':    lambda d,p: linreg(d,p),
        'FRAMA':     lambda d,p: frama(d,p),
        'JMA':       lambda d,p: jma(d,p),
        'SWMA':      lambda d,p: swma(d,p),
        'MEDIAN':    lambda d,p: median_ma(d,p),
    }

    if ma_type not in ma_functions:
        raise ValueError(f"Tipo de MA não suportado: {ma_type}")

    ma_result = ma_functions[ma_type](data, period)
    return ma_result

def run_strategy(
    candles,
    initial_capital=100.0,
    used_capital_pct=0.01,
    leverage=None,
    ma_type='SMA',
    ma_period=14,
    buy_condition=lambda prev_obv, prev_ma, current_obv, current_ma: (prev_obv > prev_ma and current_obv < current_ma),
    sell_condition=lambda prev_obv, prev_ma, current_obv, current_ma: (prev_obv < prev_ma and current_obv > current_ma),
    stop_pct=0.017,
    taker_fee=0.0004
):

    close = candles['Close'].astype('float64')
    volume = candles['Volume'].astype('float64')
    high = candles['High'].astype('float64')
    low = candles['Low'].astype('float64')

    obv = pd.Series(ta.OBV(close.values, volume.values), index=close.index)
    ma_line = create_ma_line(obv, ma_type=ma_type, period=ma_period)

    # Se leverage não é fornecida, usar 1.0 (nenhuma alavancagem)
    dynamic_leverage = (leverage is None)
    if leverage is None:
        leverage = 1.0

    position = 0
    capital = initial_capital
    trades = []

    entry_price = None
    entry_date = None
    stop_loss = None
    max_price = None
    min_price = None
    qty = None
    allocated_capital = None

    # Para acompanhamento da leverage se for dinâmica (aqui sempre 1.0 caso não informada)
    leverage_used = []

    for i in range(1, len(candles)-1):
        if np.isnan(ma_line.iloc[i]) or np.isnan(ma_line.iloc[i-1]):
            continue

        current_obv = obv.iloc[i]
        prev_obv = obv.iloc[i-1]
        current_ma = ma_line.iloc[i]
        prev_ma = ma_line.iloc[i-1]

        buy_signal = buy_condition(prev_obv, prev_ma, current_obv, current_ma)
        sell_signal = sell_condition(prev_obv, prev_ma, current_obv, current_ma)

        next_open = candles['Open'].iloc[i+1]
        current_close = candles['Close'].iloc[i]

        # Se o capital for <= 0, para de operar
        if capital <= 0:
            break

        if position == 0:
            # Alocação potencial
            potential_allocated_capital = capital * used_capital_pct * leverage
            if potential_allocated_capital <= 0:
                # Se não há capital para operar
                break

            if buy_signal:
                position = 1
                entry_date = candles.index[i+1]
                entry_price = next_open
                stop_loss = low.iloc[i+1] * (1 - stop_pct)

                allocated_capital = capital * used_capital_pct * leverage
                qty = allocated_capital / entry_price if entry_price != 0 else 0

                max_price = entry_price
                min_price = entry_price

                leverage_used.append(leverage)

            elif sell_signal:
                position = -1
                entry_date = candles.index[i+1]
                entry_price = next_open
                stop_loss = high.iloc[i+1] * (1 + stop_pct)

                allocated_capital = capital * used_capital_pct * leverage
                qty = allocated_capital / entry_price if entry_price != 0 else 0

                max_price = entry_price
                min_price = entry_price

                leverage_used.append(leverage)

        else:
            if position == 1:
                if current_close > max_price:
                    max_price = current_close
                if current_close < min_price:
                    min_price = current_close

                # Stop loss check
                if low.iloc[i+1] <= stop_loss:
                    exit_price = next_open
                    exit_date = candles.index[i+1]

                    p_l = (exit_price - entry_price) * qty
                    exit_fee = (exit_price * qty) * taker_fee
                    p_l -= exit_fee
                    capital += p_l

                    drawdown = ((entry_price - min_price) / entry_price * 100) if entry_price != 0 else 0
                    drawup = (max_price - entry_price) / entry_price * 100 if entry_price != 0 else 0
                    duration_minutes = (exit_date - entry_date).total_seconds() / 60.0

                    trades.append({
                        'Type': 'Long',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Profit (USD)': p_l,
                        'Profit (%)': (p_l / allocated_capital)*100 if allocated_capital != 0 else 0,
                        'Max Drawdown (%)': drawdown,
                        'Max Drawup (%)': drawup,
                        'Duration_minutes_raw': duration_minutes,
                        'Duration': format_duration_minutes(duration_minutes),
                        'Allocated Capital': allocated_capital,
                        'Leverage': leverage
                    })

                    position = 0
                    entry_price = None
                    entry_date = None
                    stop_loss = None
                    max_price = None
                    min_price = None
                    qty = None
                    allocated_capital = None
                    continue

                # Sinal oposto
                if sell_signal:
                    exit_price = next_open
                    exit_date = candles.index[i+1]

                    p_l = (exit_price - entry_price) * qty
                    exit_fee = (exit_price * qty) * taker_fee
                    p_l -= exit_fee
                    capital += p_l

                    drawdown = ((entry_price - min_price) / entry_price * 100) if entry_price != 0 else 0
                    drawup = (max_price - entry_price) / entry_price * 100 if entry_price != 0 else 0
                    duration_minutes = (exit_date - entry_date).total_seconds() / 60.0

                    trades.append({
                        'Type': 'Long',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Profit (USD)': p_l,
                        'Profit (%)': (p_l / allocated_capital)*100 if allocated_capital != 0 else 0,
                        'Max Drawdown (%)': drawdown,
                        'Max Drawup (%)': drawup,
                        'Duration_minutes_raw': duration_minutes,
                        'Duration': format_duration_minutes(duration_minutes),
                        'Allocated Capital': allocated_capital,
                        'Leverage': leverage
                    })

                    # Abre short
                    # Verificar capital antes de abrir nova posição
                    if capital <= 0:
                        break
                    potential_allocated_capital = capital * used_capital_pct * leverage
                    if potential_allocated_capital <= 0:
                        break

                    position = -1
                    entry_date = candles.index[i+1]
                    entry_price = next_open
                    stop_loss = high.iloc[i+1] * (1 + stop_pct)

                    allocated_capital = capital * used_capital_pct * leverage
                    qty = allocated_capital / entry_price if entry_price != 0 else 0

                    max_price = entry_price
                    min_price = entry_price

                    leverage_used.append(leverage)

            elif position == -1:
                if current_close > max_price:
                    max_price = current_close
                if current_close < min_price:
                    min_price = current_close

                # Stop loss short
                if high.iloc[i+1] >= stop_loss:
                    exit_price = next_open
                    exit_date = candles.index[i+1]

                    p_l = (entry_price - exit_price) * qty
                    exit_fee = (exit_price * qty) * taker_fee
                    p_l -= exit_fee
                    capital += p_l

                    drawdown = ((max_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
                    drawup = ((entry_price - min_price) / entry_price * 100) if entry_price != 0 else 0
                    duration_minutes = (exit_date - entry_date).total_seconds() / 60.0

                    trades.append({
                        'Type': 'Short',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Profit (USD)': p_l,
                        'Profit (%)': (p_l / allocated_capital)*100 if allocated_capital != 0 else 0,
                        'Max Drawdown (%)': drawdown,
                        'Max Drawup (%)': drawup,
                        'Duration_minutes_raw': duration_minutes,
                        'Duration': format_duration_minutes(duration_minutes),
                        'Allocated Capital': allocated_capital,
                        'Leverage': leverage
                    })

                    position = 0
                    entry_price = None
                    entry_date = None
                    stop_loss = None
                    max_price = None
                    min_price = None
                    qty = None
                    allocated_capital = None
                    continue

                # Sinal oposto (compra)
                if buy_signal:
                    exit_price = next_open
                    exit_date = candles.index[i+1]

                    p_l = (entry_price - exit_price) * qty
                    exit_fee = (exit_price * qty) * taker_fee
                    p_l -= exit_fee
                    capital += p_l

                    drawdown = ((max_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
                    drawup = ((entry_price - min_price) / entry_price * 100) if entry_price != 0 else 0
                    duration_minutes = (exit_date - entry_date).total_seconds() / 60.0

                    trades.append({
                        'Type': 'Short',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Profit (USD)': p_l,
                        'Profit (%)': (p_l / allocated_capital)*100 if allocated_capital != 0 else 0,
                        'Max Drawdown (%)': drawdown,
                        'Max Drawup (%)': drawup,
                        'Duration_minutes_raw': duration_minutes,
                        'Duration': format_duration_minutes(duration_minutes),
                        'Allocated Capital': allocated_capital,
                        'Leverage': leverage
                    })

                    # Abre Long
                    if capital <= 0:
                        break
                    potential_allocated_capital = capital * used_capital_pct * leverage
                    if potential_allocated_capital <= 0:
                        break

                    position = 1
                    entry_date = candles.index[i+1]
                    entry_price = next_open
                    stop_loss = low.iloc[i+1] * (1 - stop_pct)

                    allocated_capital = capital * used_capital_pct * leverage
                    qty = allocated_capital / entry_price if entry_price != 0 else 0

                    max_price = entry_price
                    min_price = entry_price

                    leverage_used.append(leverage)

    return trades, capital, leverage_used, dynamic_leverage

def format_number(n, suffix=""):
    return f"{n:,.2f}{suffix}"

if __name__ == '__main__':
    csv_path = 'data/Coinbase/BTC-USD/1min/coinbase_BTC-USD_1m.csv'
    timeframe_str = "210min"

    # Parâmetros de exemplo
    initial_capital = 1000.0
    used_capital_pct = 0.02
    # leverage = 2.0  # Descomente se quiser fixa
    leverage = None   # Para demonstrar o caso dinâmico (que assume 1.0)
    ma_type = 'EMA'
    ma_period = 20

    candle_cache_path = get_cache_filename_for_candles(timeframe_str)
    if os.path.exists(candle_cache_path):
        print(f"Carregando candles em cache para timeframe {timeframe_str} sem ler o CSV...")
        candles = pd.read_parquet(candle_cache_path)
    else:
        df = read_csv_if_needed(csv_path)
        candles = read_and_cache_candles(df, timeframe_str)

    trades, final_capital, leverage_used, dynamic_leverage = run_strategy(
        candles,
        initial_capital=initial_capital,
        used_capital_pct=used_capital_pct,
        leverage=leverage,
        ma_type=ma_type,
        ma_period=ma_period
    )

    if trades:
        df_trades = pd.DataFrame(trades)

        # Cálculo do Buy & Hold
        first_price = candles['Open'].iloc[0]
        last_price = candles['Close'].iloc[-1]
        bh_qty = initial_capital / first_price
        bh_final_equity = bh_qty * last_price
        bh_profit = bh_final_equity - initial_capital
        bh_profit_pct = (bh_profit / initial_capital)*100 if initial_capital != 0 else 0

        total_trades = len(df_trades)
        total_profit = df_trades['Profit (USD)'].sum()
        profits = df_trades['Profit (USD)']

        winning_trades = (profits > 0).sum()
        losing_trades = (profits < 0).sum()
        break_even_trades = (profits == 0).sum()

        num_long = (df_trades['Type'] == 'Long').sum()
        num_short = (df_trades['Type'] == 'Short').sum()

        max_profit_usd = df_trades['Profit (USD)'].max()
        min_profit_usd = df_trades['Profit (USD)'].min()
        max_profit_row = df_trades[df_trades['Profit (USD)'] == max_profit_usd].iloc[0]
        min_profit_row = df_trades[df_trades['Profit (USD)'] == min_profit_usd].iloc[0]

        max_profit_pct = max_profit_row['Profit (%)']
        min_profit_pct = min_profit_row['Profit (%)']

        avg_profit_usd = total_profit / total_trades if total_trades > 0 else 0
        avg_profit_pct = df_trades['Profit (%)'].mean() if total_trades > 0 else 0

        long_trades_df = df_trades[df_trades['Type'] == 'Long']
        short_trades_df = df_trades[df_trades['Type'] == 'Short']

        def safe_stat(series, func):
            return func(series) if not series.empty else None

        max_dd_long = safe_stat(long_trades_df['Max Drawdown (%)'], np.max)
        avg_dd_long = safe_stat(long_trades_df['Max Drawdown (%)'], np.mean)

        max_dd_short = safe_stat(short_trades_df['Max Drawdown (%)'], np.max)
        avg_dd_short = safe_stat(short_trades_df['Max Drawdown (%)'], np.mean)

        durations_raw = df_trades['Duration_minutes_raw']
        if total_trades > 0:
            longest_duration = durations_raw.max()
            shortest_duration = durations_raw.min()
            avg_duration = durations_raw.mean()

            longest_str = format_duration_minutes(longest_duration)
            shortest_str = format_duration_minutes(shortest_duration)
            avg_str = format_duration_minutes(avg_duration)
        else:
            longest_str = "00:00:00:00:00"
            shortest_str = "00:00:00:00:00"
            avg_str = "00:00:00:00:00"

        def val_pct(value):
            return f"{value:,.2f}%" if value is not None else "N/A"

        # Total Profit % em relação ao capital inicial
        total_profit_pct = (total_profit / initial_capital) * 100 if initial_capital != 0 else 0

        strategy_vs_bh_usd = (final_capital - bh_final_equity)
        strategy_vs_bh_pct = (strategy_vs_bh_usd / abs(bh_profit)*100) if bh_profit != 0 else (100 if strategy_vs_bh_usd > 0 else 0)

        def pct_of_total(n):
            return val_pct((n/total_trades)*100 if total_trades > 0 else 0)

        summary_data = [
            ["Total Trades", f"{total_trades}", "100.00%"],
            ["Long Trades", f"{num_long}", pct_of_total(num_long)],
            ["Short Trades", f"{num_short}", pct_of_total(num_short)],
            ["Winning Trades", f"{winning_trades}", pct_of_total(winning_trades)],
            ["Losing Trades", f"{losing_trades}", pct_of_total(losing_trades)],
            ["Break-even Trades", f"{break_even_trades}", pct_of_total(break_even_trades)],
            ["Total Profit", f"{format_number(total_profit, ' USD')}", val_pct(total_profit_pct)],
            ["Avg Profit per Trade", f"{format_number(avg_profit_usd, ' USD')}", val_pct(avg_profit_pct)],
            ["Max Profit (per trade)", f"{format_number(max_profit_usd, ' USD')}", val_pct(max_profit_pct)],
            ["Min Profit (per trade)", f"{format_number(min_profit_usd, ' USD')}", val_pct(min_profit_pct)],
            ["Max Drawdown (Long)", "" if max_dd_long is not None else "N/A", val_pct(max_dd_long)],
            ["Average Drawdown (Long)", "" if avg_dd_long is not None else "N/A", val_pct(avg_dd_long)],
            ["Max Drawdown (Short)", "" if max_dd_short is not None else "N/A", val_pct(max_dd_short)],
            ["Average Drawdown (Short)", "" if avg_dd_short is not None else "N/A", val_pct(avg_dd_short)],
            ["Longest Operation Time", longest_str, "N/A"],
            ["Shortest Operation Time", shortest_str, "N/A"],
            ["Average Operation Time", avg_str, "N/A"],
            ["Buy and Hold Profit", f"{format_number(bh_profit, ' USD')}", val_pct(bh_profit_pct)],
            ["Strategy vs B&H", f"{format_number(strategy_vs_bh_usd, ' USD')}", val_pct(strategy_vs_bh_pct)]
        ]

        print("\nSummary:")
        print(tabulate(summary_data, headers=["Metric", "Value (Abs)", "Value (%)"], tablefmt="grid", showindex=False))

        # Most Representative Trades com formatação
        def format_trade_row(row):
            return {
                'Type': row['Type'],
                'Entry Date': row['Entry Date'],
                'Entry Price': format_number(row['Entry Price'], ''),
                'Exit Date': row['Exit Date'],
                'Exit Price': format_number(row['Exit Price'], ''),
                'Profit (USD)': format_number(row['Profit (USD)'], ''),
                'Profit (%)': val_pct(row['Profit (%)'])
            }

        best_trade_formatted = format_trade_row(max_profit_row)
        worst_trade_formatted = format_trade_row(min_profit_row)

        print("\nMost Representative Trades:")

        print("\nBest Trade:")
        print(tabulate(pd.DataFrame([best_trade_formatted]), headers="keys", tablefmt="grid", showindex=False))

        print("\nWorst Trade:")
        print(tabulate(pd.DataFrame([worst_trade_formatted]), headers="keys", tablefmt="grid", showindex=False))

        # Se a alavancagem não foi informada (dinâmica), mostrar max, avg, min
        if dynamic_leverage and leverage_used:
            max_lev = np.max(leverage_used)
            min_lev = np.min(leverage_used)
            avg_lev = np.mean(leverage_used)
            print("\nLeverage Statistics (Dynamic):")
            print(f"Max Leverage: {max_lev:.2f}x")
            print(f"Min Leverage: {min_lev:.2f}x")
            print(f"Average Leverage: {avg_lev:.2f}x")

    else:
        print("Nenhuma trade foi realizada pela estratégia.")