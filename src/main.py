import time
import pandas as pd
from tabulate import tabulate
import re
import talib as ta
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Definição de todas as funções no nível superior

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
        'm': 'M',
        'y': 'Y'
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
    print(f"Iniciando resampling com frequência: {pandas_tf}")

    df = df.set_index("Date").sort_index()
    candles = df.resample(pandas_tf).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna(how='any')

    print("Resampling concluído.")
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
    wts = wts / np.sum(wts)
    res = data.rolling(period).apply(lambda x: np.sum(x * wts), raw=True)
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

    try:
        ma_result = ma_functions[ma_type](data, period)
    except Exception as e:
        ma_result = np.full(len(data), np.nan)

    if not isinstance(ma_result, pd.Series):
        ma_result = pd.Series(ma_result, index=data.index)

    return ma_result

def run_strategy(
    candles,
    initial_capital=100.0,
    used_capital_pct=0.4,
    leverage=7,  # Alavancagem fixa
    ma_type='SMA',
    ma_period=2,
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

    position = 0  # 0: sem posição, 1: Long, -1: Short
    capital = initial_capital
    trades = []
    liquidations = 0  # Contador de liquidações

    entry_price = None
    entry_date = None
    stop_loss = None
    qty = None
    allocated_capital = None

    # Variáveis para Trailing Stop
    trailing_step = 0  # Incremento de 1% de lucro
    last_trailing_update = 0  # Último nível de lucro que atualizou o stop loss

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

        if capital <= 0:
            break

        if position == 0:
            if buy_signal or sell_signal:
                potential_allocated_capital = capital * used_capital_pct * leverage
                if potential_allocated_capital <= 0:
                    continue

                if buy_signal:
                    position = 1
                    entry_date = candles.index[i+1]
                    entry_price = next_open
                    stop_loss = low.iloc[i+1] * (1 - stop_pct)
                    allocated_capital = potential_allocated_capital
                    qty = allocated_capital / entry_price if entry_price != 0 else 0
                    max_price = entry_price
                    min_price = entry_price

                    # Deduzir a taxa de corretagem na entrada
                    entry_fee = allocated_capital * taker_fee
                    capital -= entry_fee

                    # Resetar Trailing Stop Variables
                    trailing_step = 0
                    last_trailing_update = 0

                elif sell_signal:
                    position = -1
                    entry_date = candles.index[i+1]
                    entry_price = next_open
                    stop_loss = high.iloc[i+1] * (1 + stop_pct)
                    allocated_capital = potential_allocated_capital
                    qty = allocated_capital / entry_price if entry_price != 0 else 0
                    max_price = entry_price
                    min_price = entry_price

                    # Deduzir a taxa de corretagem na entrada
                    entry_fee = allocated_capital * taker_fee
                    capital -= entry_fee

                    # Resetar Trailing Stop Variables
                    trailing_step = 0
                    last_trailing_update = 0

        else:
            if position == 1:
                # Atualizar preços máximo e mínimo para monitorar drawdown
                max_price = max(max_price, current_close)
                min_price = min(min_price, candles['Low'].iloc[i])

                # Definir drawdown e drawup antes das condições
                drawdown = ((entry_price - min_price) / entry_price * 100) if entry_price != 0 else 0
                drawup = (max_price - entry_price) / entry_price * 100 if entry_price != 0 else 0

                # Calcular lucro atual
                current_profit_pct = ((current_close - entry_price) / entry_price) * 100 if entry_price != 0 else 0

                # Verificar se o lucro atingiu um novo nível de trailing stop
                while current_profit_pct >= (last_trailing_update + 1):
                    trailing_step += 1
                    last_trailing_update += 1
                    # Mover o stop loss para cima em 1%
                    stop_loss = entry_price * (1 + (trailing_step * 0.01))

                # Verificar se o stop loss foi atingido
                if low.iloc[i+1] <= stop_loss:
                    exit_price = next_open
                    exit_date = candles.index[i+1]

                    p_l = (exit_price - entry_price) * qty

                    # Deduzir a taxa de corretagem na saída
                    exit_fee = allocated_capital * taker_fee
                    p_l -= exit_fee

                    capital += p_l

                    # Verificar liquidação
                    if capital < 0:
                        liquidations += 1
                        capital = 0  # Resetar capital
                        p_l = -allocated_capital  # Perda total da posição

                    duration_minutes = (exit_date - entry_date).total_seconds() / 60.0

                    trades.append({
                        'Type': 'Long',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Profit (USD)': p_l,
                        'Profit (%)': (p_l / allocated_capital) * 100 if allocated_capital != 0 else 0,  # Relativo ao capital alocado
                        'Max Drawdown (%)': drawdown,
                        'Max Drawup (%)': drawup,
                        'Duration_minutes_raw': duration_minutes,
                        'Duration': format_duration_minutes(duration_minutes),
                        'Allocated Capital': allocated_capital,
                        'Leverage': leverage  # Alavancagem fixa
                    })

                    position = 0
                    entry_price = None
                    entry_date = None
                    stop_loss = None
                    max_price = None
                    min_price = None
                    qty = None
                    allocated_capital = None
                    trailing_step = 0
                    last_trailing_update = 0
                    continue

                # Verificar sinal de inversão para fechar posição Long e abrir Short
                if sell_signal:
                    exit_price = next_open
                    exit_date = candles.index[i+1]

                    p_l = (exit_price - entry_price) * qty

                    # Deduzir a taxa de corretagem na saída
                    exit_fee = allocated_capital * taker_fee
                    p_l -= exit_fee

                    capital += p_l

                    duration_minutes = (exit_date - entry_date).total_seconds() / 60.0

                    trades.append({
                        'Type': 'Long',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Profit (USD)': p_l,
                        'Profit (%)': (p_l / allocated_capital) * 100 if allocated_capital != 0 else 0,  # Relativo ao capital alocado
                        'Max Drawdown (%)': drawdown,
                        'Max Drawup (%)': drawup,
                        'Duration_minutes_raw': duration_minutes,
                        'Duration': format_duration_minutes(duration_minutes),
                        'Allocated Capital': allocated_capital,
                        'Leverage': leverage  # Alavancagem fixa
                    })

                    # Abrir posição Short
                    position = -1
                    entry_date = candles.index[i+1]
                    entry_price = next_open
                    stop_loss = high.iloc[i+1] * (1 + stop_pct)
                    allocated_capital = capital * used_capital_pct * leverage
                    qty = allocated_capital / entry_price if entry_price != 0 else 0

                    # Deduzir a taxa de corretagem na entrada da nova posição
                    entry_fee = allocated_capital * taker_fee
                    capital -= entry_fee

                    max_price = entry_price
                    min_price = entry_price

                    # Resetar Trailing Stop Variables
                    trailing_step = 0
                    last_trailing_update = 0
                    continue

            elif position == -1:
                # Atualizar preços máximo e mínimo para monitorar drawdown
                max_price = max(max_price, current_close)
                min_price = min(min_price, candles['Low'].iloc[i])

                # Definir drawdown e drawup antes das condições
                drawdown = ((max_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
                drawup = ((entry_price - min_price) / entry_price * 100) if entry_price != 0 else 0

                # Calcular lucro atual
                current_profit_pct = ((entry_price - current_close) / entry_price) * 100 if entry_price != 0 else 0

                # Verificar se o lucro atingiu um novo nível de trailing stop
                while current_profit_pct >= (last_trailing_update + 1):
                    trailing_step += 1
                    last_trailing_update += 1
                    # Mover o stop loss para baixo em 1%
                    stop_loss = entry_price * (1 - (trailing_step * 0.01))

                # Verificar se o stop loss foi atingido
                if high.iloc[i+1] >= stop_loss:
                    exit_price = next_open
                    exit_date = candles.index[i+1]

                    p_l = (entry_price - exit_price) * qty

                    # Deduzir a taxa de corretagem na saída
                    exit_fee = allocated_capital * taker_fee
                    p_l -= exit_fee

                    capital += p_l

                    # Verificar liquidação
                    if capital < 0:
                        liquidations += 1
                        capital = 0  # Resetar capital
                        p_l = -allocated_capital  # Perda total da posição

                    duration_minutes = (exit_date - entry_date).total_seconds() / 60.0

                    trades.append({
                        'Type': 'Short',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Profit (USD)': p_l,
                        'Profit (%)': (p_l / allocated_capital) * 100 if allocated_capital != 0 else 0,  # Relativo ao capital alocado
                        'Max Drawdown (%)': drawdown,
                        'Max Drawup (%)': drawup,
                        'Duration_minutes_raw': duration_minutes,
                        'Duration': format_duration_minutes(duration_minutes),
                        'Allocated Capital': allocated_capital,
                        'Leverage': leverage  # Alavancagem fixa
                    })

                    position = 0
                    entry_price = None
                    entry_date = None
                    stop_loss = None
                    max_price = None
                    min_price = None
                    qty = None
                    allocated_capital = None
                    trailing_step = 0
                    last_trailing_update = 0
                    continue

                # Verificar sinal de inversão para fechar posição Short e abrir Long
                if buy_signal:
                    exit_price = next_open
                    exit_date = candles.index[i+1]

                    p_l = (entry_price - exit_price) * qty

                    # Deduzir a taxa de corretagem na saída
                    exit_fee = allocated_capital * taker_fee
                    p_l -= exit_fee

                    capital += p_l

                    duration_minutes = (exit_date - entry_date).total_seconds() / 60.0

                    trades.append({
                        'Type': 'Short',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Profit (USD)': p_l,
                        'Profit (%)': (p_l / allocated_capital) * 100 if allocated_capital != 0 else 0,  # Relativo ao capital alocado
                        'Max Drawdown (%)': drawdown,
                        'Max Drawup (%)': drawup,
                        'Duration_minutes_raw': duration_minutes,
                        'Duration': format_duration_minutes(duration_minutes),
                        'Allocated Capital': allocated_capital,
                        'Leverage': leverage  # Alavancagem fixa
                    })

                    # Abrir posição Long
                    position = 1
                    entry_date = candles.index[i+1]
                    entry_price = next_open
                    stop_loss = low.iloc[i+1] * (1 - stop_pct)
                    allocated_capital = capital * used_capital_pct * leverage
                    qty = allocated_capital / entry_price if entry_price != 0 else 0

                    # Deduzir a taxa de corretagem na entrada da nova posição
                    entry_fee = allocated_capital * taker_fee
                    capital -= entry_fee

                    max_price = entry_price
                    min_price = entry_price

                    # Resetar Trailing Stop Variables
                    trailing_step = 0
                    last_trailing_update = 0
                    continue

    # Retornar também o número de liquidações
    return trades, capital, liquidations

def test_ma(ma, candles, initial_capital, used_capital_pct, leverage, taker_fee, periods):
    best_ma_period = None
    best_total_profit = -np.inf
    best_result_for_this_ma = None
    consecutive_worse = 0
    last_total_profit = -np.inf
    tested_periods = 0

    for ma_period in periods:
        trades, final_capital, liquidations = run_strategy(
            candles,
            initial_capital=initial_capital,
            used_capital_pct=used_capital_pct,
            leverage=leverage,
            ma_type=ma,
            ma_period=ma_period,
            taker_fee=taker_fee
        )
        tested_periods += 1  # Incrementa o contador de períodos testados

        if len(trades) > 0:
            df_trades = pd.DataFrame(trades)
            if 'Profit (USD)' in df_trades.columns:
                total_profit = df_trades['Profit (USD)'].sum()
            else:
                total_profit = 0
        else:
            total_profit = 0

        if np.isinf(total_profit) or np.isnan(total_profit):
            continue

        if total_profit > best_total_profit:
            best_total_profit = total_profit
            best_ma_period = ma_period
            best_result_for_this_ma = (ma, ma_period, trades, final_capital, best_total_profit, liquidations)
            consecutive_worse = 0
        else:
            if total_profit < last_total_profit:
                consecutive_worse += 1

        last_total_profit = total_profit

        if consecutive_worse > 5:
            break

    if best_result_for_this_ma is None:
        best_result_for_this_ma = (ma, None, [], initial_capital, 0, 0)

    # Retorna também quantos períodos foram testados
    return best_result_for_this_ma, tested_periods

def format_number(n, suffix=""):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    return f"{n:,.2f}{suffix}"

def val_pct(value):
    if isinstance(value, (int, float)):
        return f"{value:,.2f}%"
    return "N/A"

# Protege a execução principal
if __name__ == '__main__':
    # Parâmetros iniciais
    csv_path = 'data/Coinbase/BTC-USD/1min/coinbase_BTC-USD_1m.csv'
    timeframe_str = "25min"  
    initial_capital = 100.0
    used_capital_pct = 0.02  
    leverage = 20  
    taker_fee = 0.0004  

    ma_types = [
        'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3',
        'HT_TRENDLINE', 'RMA', 'ZLEMA', 'HMA', 'ALMA', 'WILDERS', 'LINREG',
        'MEDIAN', 'FRAMA', 'JMA', 'SWMA'
    ]

    periods = range(2, 21)  # Períodos de 2 a 20
    number_of_periods = len(periods)
    total_steps = len(ma_types)  # Barra de progresso baseada no número de MAs

    candle_cache_path = get_cache_filename_for_candles(timeframe_str)
    if os.path.exists(candle_cache_path):
        print("Carregando candles do cache...")
        candles = pd.read_parquet(candle_cache_path)
    else:
        df = read_csv_if_needed(csv_path)
        candles = read_and_cache_candles(df, timeframe_str)

    # Limitar o volume de dados para testes (opcional)
    # candles = candles.iloc[:10000]  # Descomente para limitar

    print(f"\n\033[94mExecutando testes com diferentes MAs e períodos...\033[0m")

    overall_results = []
    tested_periods_total = 0

    # Limitar o número de processos para evitar sobrecarga
    max_workers = os.cpu_count() or 1  # Usa 1 se os.cpu_count() retornar None
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submeter tarefas para cada MA
        futures = {executor.submit(test_ma, ma, candles, initial_capital, used_capital_pct, leverage, taker_fee, periods): ma for ma in ma_types}
        
        # Barra de progresso baseada no número de MAs
        with tqdm(total=total_steps, desc="Processando MAs") as pbar:
            for future in as_completed(futures):
                ma = futures[future]
                try:
                    res, tested = future.result(timeout=300)  # Timeout de 5 minutos por tarefa
                    overall_results.append(res)
                    pbar.update(1)  # Incrementa a barra por MA completada
                    tested_periods_total += tested
                except Exception as e:
                    print(f"Erro ao processar MA {ma}: {e}")
                    pbar.update(1)  # Atualiza a barra mesmo em caso de erro

    if not overall_results:
        print("Nenhum resultado foi obtido. Verifique as entradas e a lógica da estratégia.")
        exit()

    # Encontrar o melhor resultado global baseado no Total Profit
    best_global_result = max(overall_results, key=lambda x: x[4])
    best_ma = best_global_result[0]
    best_ma_period = best_global_result[1]
    trades = best_global_result[2]
    final_capital = best_global_result[3]
    total_profit = best_global_result[4]
    total_liquidations = best_global_result[5]

    df_trades = pd.DataFrame(trades)

    # Cálculo B&H líquido com alavancagem igual à estratégia para comparação justa
    bh_allocated_capital = initial_capital * used_capital_pct * leverage
    bh_entry_fee = bh_allocated_capital * taker_fee
    bh_capital_after_entry = bh_allocated_capital - bh_entry_fee
    first_price = candles['Open'].iloc[0]
    bh_qty = bh_capital_after_entry / first_price if first_price != 0 else 0
    last_price = candles['Close'].iloc[-1]
    bh_final_equity = bh_qty * last_price
    bh_exit_fee = bh_final_equity * taker_fee
    bh_final_equity_net = bh_final_equity - bh_exit_fee

    # **Correção 1: Definir Buy and Hold Profit em relação ao capital inicial**
    bh_profit = bh_final_equity_net - initial_capital
    bh_profit_pct = (bh_profit / initial_capital) * 100 if initial_capital != 0 else 0  # Relativo ao capital inicial

    # **Correção 2: Calcular Strategy vs B&H com base no Buy and Hold Profit**
    strategy_vs_bh_usd = total_profit - bh_profit
    strategy_vs_bh_pct = (strategy_vs_bh_usd / bh_profit) * 100 if bh_profit != 0 else "N/A"

    # Cálculo de Lucro Total Percentual Relativo ao Capital Inicial
    total_profit_pct = (total_profit / initial_capital) * 100 if initial_capital != 0 else 0

    total_trades = len(df_trades)
    if total_trades > 0 and 'Profit (USD)' in df_trades.columns:
        profits = df_trades['Profit (USD)']
    else:
        profits = pd.Series(dtype='float64')

    winning_trades = (profits > 0).sum()
    losing_trades = (profits < 0).sum()
    break_even_trades = (profits == 0).sum()

    num_long = (df_trades['Type'] == 'Long').sum() if total_trades > 0 else 0
    num_short = (df_trades['Type'] == 'Short').sum() if total_trades > 0 else 0

    if total_trades > 0 and 'Profit (USD)' in df_trades.columns:
        max_profit_usd = df_trades['Profit (USD)'].max()
        min_profit_usd = df_trades['Profit (USD)'].min()
        max_profit_row = df_trades[df_trades['Profit (USD)'] == max_profit_usd].iloc[0]
        min_profit_row = df_trades[df_trades['Profit (USD)'] == min_profit_usd].iloc[0]

        max_profit_pct = max_profit_row['Profit (%)']
        min_profit_pct = min_profit_row['Profit (%)']
    else:
        max_profit_usd = 0
        min_profit_usd = 0
        max_profit_pct = 0
        min_profit_pct = 0

    avg_profit_usd = total_profit / total_trades if total_trades > 0 else 0
    avg_profit_pct = df_trades['Profit (%)'].mean() if (total_trades > 0 and 'Profit (%)' in df_trades.columns) else 0

    # Cálculo de Drawdown e Duração
    if total_trades > 0:
        long_trades_df = df_trades[df_trades['Type'] == 'Long']
        short_trades_df = df_trades[df_trades['Type'] == 'Short']

        max_dd_long = long_trades_df['Max Drawdown (%)'].max() if not long_trades_df.empty else "N/A"
        avg_dd_long = long_trades_df['Max Drawdown (%)'].mean() if not long_trades_df.empty else "N/A"

        max_dd_short = short_trades_df['Max Drawdown (%)'].max() if not short_trades_df.empty else "N/A"
        avg_dd_short = short_trades_df['Max Drawdown (%)'].mean() if not short_trades_df.empty else "N/A"

        durations_raw = df_trades['Duration_minutes_raw'] if ('Duration_minutes_raw' in df_trades.columns and total_trades > 0) else pd.Series([])
        if not durations_raw.empty:
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
    else:
        max_dd_long = avg_dd_long = max_dd_short = avg_dd_short = "N/A"
        longest_str = shortest_str = avg_str = "00:00:00:00:00"

    # Função para formatar porcentagens de trades
    def pct_of_total(n):
        return val_pct((n / total_trades) * 100 if total_trades > 0 else 0)

    summary_data = [
        ["Timeframe", timeframe_str, "N/A"],
        ["Best MA Type", f"{best_ma}", "N/A"],
        ["Best MA Period", f"{best_ma_period}", "N/A"],
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
        ["Max Drawdown (Long)", f"{format_number(max_dd_long, '%')}" if max_dd_long != "N/A" else "N/A", "N/A"],
        ["Average Drawdown (Long)", f"{format_number(avg_dd_long, '%')}" if avg_dd_long != "N/A" else "N/A", "N/A"],
        ["Max Drawdown (Short)", f"{format_number(max_dd_short, '%')}" if max_dd_short != "N/A" else "N/A", "N/A"],
        ["Average Drawdown (Short)", f"{format_number(avg_dd_short, '%')}" if avg_dd_short != "N/A" else "N/A", "N/A"],
        ["Longest Operation Time", longest_str, "N/A"],
        ["Shortest Operation Time", shortest_str, "N/A"],
        ["Average Operation Time", avg_str, "N/A"],
        ["Buy and Hold Profit", f"{format_number(bh_profit, ' USD')}", val_pct(bh_profit_pct)],
        ["Strategy vs B&H", f"{format_number(strategy_vs_bh_usd, ' USD')}", val_pct(strategy_vs_bh_pct)],  # Linha corrigida
        ["Leverage (Fixed)", f"{leverage}x", "N/A"],
        ["Total Liquidations", f"{total_liquidations}", val_pct((total_liquidations / total_trades) * 100 if total_trades > 0 else 0)],
        ["Long Liquidations", "0", "0.00%"],  # Implementar se necessário
        ["Short Liquidations", "0", "0.00%"],  # Implementar se necessário
    ]

    print("\n\033[93mSummary:\033[0m")
    print(tabulate(summary_data, headers=["Metric", "Value (Abs)", "Value (%)"], tablefmt="grid", showindex=False))

    def format_trade_row(row):
        return {
            'Type': row.get('Type', 'N/A'),
            'Entry Date': row.get('Entry Date', 'N/A'),
            'Entry Price': format_number(row.get('Entry Price', None), ''),
            'Exit Date': row.get('Exit Date', 'N/A'),
            'Exit Price': format_number(row.get('Exit Price', None), ''),
            'Profit (USD)': format_number(row.get('Profit (USD)', None), ''),
            'Profit (%)': val_pct(row.get('Profit (%)', None)),
            'Leverage': f"{leverage}x"  # Alavancagem fixa
        }

    if total_trades > 0:
        max_profit_row_dict = dict(max_profit_row) if total_trades > 0 else {}
        min_profit_row_dict = dict(min_profit_row) if total_trades > 0 else {}

        best_trade_formatted = format_trade_row(max_profit_row_dict) if max_profit_row_dict else {}
        worst_trade_formatted = format_trade_row(min_profit_row_dict) if min_profit_row_dict else {}

        print("\n\033[94mMost Representative Trades:\033[0m")

        if best_trade_formatted:
            print("\n\033[92mBest Trade:\033[0m")
            print(tabulate(pd.DataFrame([best_trade_formatted]), headers="keys", tablefmt="grid", showindex=False))

        if worst_trade_formatted:
            print("\n\033[91mWorst Trade:\033[0m")
            print(tabulate(pd.DataFrame([worst_trade_formatted]), headers="keys", tablefmt="grid", showindex=False))
    else:
        print("\nNenhuma trade foi realizada na melhor configuração encontrada.")