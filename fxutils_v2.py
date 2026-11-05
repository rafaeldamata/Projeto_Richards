import pandas as pd
import numpy as np
import duckdb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit


def PlotDataBase(df, args, coluna='Observed', tamanho_janela=20):
    """
    Plota dois subplots:
    1. Dados acumulados e a projeção acumulada de Richards.
    2. Dados observados (diferença) e a derivada da projeção de Richards (np.diff).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    TRAIN_DATA = args.train_data
    WINDOW = args.window_size
    
    Coluna = df[coluna].values.astype(float)
    
    # Transformação dos dados originais
    Coluna = Coluna - 1
    Coluna = Coluna * 100000
    Coluna = Coluna - Coluna[0]

    y_data = np.cumsum(Coluna)
    x_data = np.arange(1, len(y_data) + 1)

    StartIndex = 0
    FinalIndex = TRAIN_DATA

    x_train_data = x_data[StartIndex:FinalIndex+StartIndex]
    y_train_data = y_data[StartIndex:FinalIndex+StartIndex]

    # Chutes iniciais mais seguros baseados nos dados de treino
    y_min, y_max = np.min(y_train_data), np.max(y_train_data)
    y_range = y_max - y_min if y_max != y_min else 1.0
    
    p0 = [y_min, y_max + y_range, 0.1, np.mean(x_train_data), 0.8] 
    
    # Limites mais amplos mas seguros para evitar falhas de convergência
    bounds = (
        [-np.inf, -np.inf, 0.0, 0.0, 0.001], 
        [np.inf, np.inf, 10.0, np.max(x_data)*2, 100.0]
    )
    maxfev = 9000000

    try:
        poptN, _ = curve_fit(Richards, x_train_data, y_train_data, p0=p0, bounds=bounds, maxfev=maxfev)
        
        # -------------------------------------------------
        y_pred_train = Richards(x_train_data, *poptN)
        # Importante: certifique-se de que Adjustment_Metrics está definida no seu escopo
        sse_train, rmse_train, r2_train = Adjustment_Metrics(y_train_data, y_pred_train)
        
        Normal_y_predicted = Richards(x_data, *poptN)
        sse_acc, rmse_acc, r2_acc = Adjustment_Metrics(y_data, Normal_y_predicted)
        
        print(f"Parâmetros otimizados: {poptN}")
        print("--------------------") 
        print(f"R² Treino: {r2_train:.3f}")
        print(f"R² Normal (Acumulado): {r2_acc:.3f}")
        print(f"RMSE Treino: {rmse_train:.3f}")
        print(f"RMSE Normal: {rmse_acc:.3f}")
        print("--------------------")

        # SUBPLOT 1: Dados Acumulados vs Projeção Acumulada de Richards
        ax1.plot(x_data, y_data, color='lightgray', linestyle='-', linewidth=1.5, label='Dados Acumulados')
        ax1.plot(x_data, Normal_y_predicted, color='royalblue', linestyle='-', linewidth=2, label='Projeção Richards (Acumulada)')
        ax1.axvline(x=TRAIN_DATA, color='orange', linestyle='--', linewidth=2, label=f'Limite de Treino ({TRAIN_DATA})')
        ax1.set_title(f'Dados Acumulados | Window: {WINDOW} | Train Limit: {TRAIN_DATA}')
        ax1.set_ylabel('Valor Acumulado')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # SUBPLOT 2: Dados Observados vs np.diff(Projeção Richards)
        obs_diff = np.diff(y_data, prepend=y_data[0])
        pred_diff = np.diff(Normal_y_predicted, prepend=Normal_y_predicted[0])

        ax2.plot(x_data, obs_diff, color='lightgray', linestyle='-', linewidth=1.5, label='Dados Observados (Diff)')
        ax2.plot(x_data, pred_diff, color='red', linestyle='-', linewidth=2, label='np.diff(Projeção Richards)')
        ax2.axvline(x=TRAIN_DATA, color='orange', linestyle='--', linewidth=2, label=f'Limite de Treino ({TRAIN_DATA})')
        ax2.set_title(f'Dados Observados vs Derivada da Projeção de Richards')
        ax2.set_ylabel('Preço / Variação')
        ax2.set_xlabel('Índice / Tempo')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro no ajuste da Curva de Richards na janela: {e}")

    
    
    # Cria a figura com um bom tamanho para acomodar todos os subgráficos
    
def PlotSimulation(dy_normal,dy_fit):
    if len(dy_normal) != len(dy_fit):
        raise ValueError("Os vetores devem ter o mesmo comprimento.")
        
    # Eixo X variando de 1 até o tamanho de vetor_y
    x = range(1, len(dy_normal) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Configuração do Eixo Y Primário (Vetor Y)
    color_y = 'tab:blue'
    ax1.set_xlabel('Índice (1 a N)', fontsize=12)
    ax1.set_ylabel('Real Data', color=color_y, fontsize=12)
    line1 = ax1.plot(x, dy_normal, color=color_y, label='Real', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_y)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # Configuração do Eixo Y Secundário
    ax2 = ax1.twinx()  
    color_z = 'tab:orange'
    ax2.set_ylabel('Simulated Data', color=color_z, fontsize=12)
    line2 = ax2.plot(x, dy_fit, color=color_z, linestyle='--', label='Simulated', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_z)
    
    # Unificando as legendas dos dois eixos em um único local
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Real data Vs Simulated data)', fontsize=13, pad=15)
    fig.tight_layout()
    plt.show()



def ReadDataFromDataBase(db_parametres):
    con = duckdb.connect(db_parametres['Path'])
    #--------------------------------------------------------------------
    tf_mapping = {'Tick-By-Tick': None, '1M': '1 minute', '5M': '5 minutes', '15M': '15 minutes',
                  '30M': '30 minutes', '1H': '1 hour', '4H': '4 hours', '1D': '1 day', 
                  '1W': '1 week', '1MW': '1 month'}
    interval_sql = tf_mapping.get(db_parametres['dtframe'], '')
    #--------------------------------------------------------------------
    if db_parametres['dtframe'] == 'Tick-By-Tick':
        query = f"SELECT timestamp AS Date, {db_parametres['dtserie']} AS val FROM forex_ticks WHERE pair = 'EURUSD' ORDER BY timestamp ASC LIMIT {db_parametres['dtsize']};"
    else:
        query = f"""
        WITH ohlc AS (
            SELECT time_bucket(INTERVAL '{interval_sql}', timestamp) AS Date,
                FIRST({db_parametres['dtserie']}) AS Open, MAX({db_parametres['dtserie']}) AS High,
                MIN({db_parametres['dtserie']}) AS Low, LAST({db_parametres['dtserie']}) AS Close
            FROM forex_ticks WHERE pair = 'EURUSD' GROUP BY Date ORDER BY Date ASC
        )
        SELECT Date, {db_parametres['dttype']} AS val FROM ohlc LIMIT {db_parametres['dtsize']};
        """
    #--------------------------------------------------------------------
    df = con.execute(query).df()
    con.close()
    obs_ajustado = df['val'].to_numpy(dtype=float) - 1.0
    rdf = pd.DataFrame({
        'Date': df['Date'].values,
        'Observed': obs_ajustado,
        'Accumulated': np.cumsum(obs_ajustado)
    })
    return rdf
    
def Richards(t, A, K, B, Q, nu):
    # Protege contra overflow no expoente da exponencial
    t_clipped = np.clip(t, -500, 500)
    exp_term = np.exp(-B * t_clipped)
    # Base da curva de Richards
    base = 1.0 + Q * exp_term
    # Expoente correto (1.0 / nu), protegido contra divisão por zero
    inv_nu = 1.0 / np.maximum(nu, 1e-5)
    return A + (K - A) / np.power(base, inv_nu)
    
def Adjustment_Metrics(y_true, y_pred, num_params=5):
    """
    Calcula SSE, RMSE e R² entre os dados reais e a predição.
    """
    # Remover eventuais NaNs ou infinitos para evitar erros
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    # SSE (Soma dos Erros Quadráticos)
    sse = np.sum((y_true - y_pred) ** 2)
    
    # RMSE (Raiz do Erro Quadrático Médio)
    rmse = np.sqrt(sse / len(y_true))
    
    # R² (Coeficiente de Determinação)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_total == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - (sse / ss_total)
        
    return sse, rmse, r2

def PlotResults(data, Window, TrainLimit, InitIndex, FinalIndex, n_prediction, nr2, vr2, filename):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    y_global    = data
    t_global    = np.arange(1, len(y_global) + 1)

    # -----------------------------------------
    # PAINEL 1: Série Global + Janela + Acumulado
    # -----------------------------------------
    ax1.plot(t_global, np.diff(y_global, prepend=0), color='lightgray', linestyle='-', linewidth=1.5, label='Data')
    ax1.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limit')
    ax1.plot(t_global, np.diff(n_prediction, prepend=0), color='red', linestyle='-', linewidth=1.5, label='Data')
    
    ax1.set_title(f'Observed data: Window:{Window} | Train Limit:{TrainLimit})')
    ax1.set_ylabel(f'Preço ')
    ax1.grid(True, alpha=0.3)
    # -----------------------------------------
    ax2.plot(t_global, y_global, color='lightgray', linestyle='-', linewidth=1.5, label='Accumulated data')
    ax2.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limit')
    ax2.plot(t_global, n_prediction, color='royalblue', linestyle='-', linewidth=1.5, label='Accumulated data')
    
    ax2.set_title(f'Accumulated data: Window:{Window} | Train Limit:{TrainLimit})')
    ax2.set_ylabel(f'Value ')
    ax2.grid(True, alpha=0.3)
    # -----------------------------------------
    ax3.plot(t_global, y_global, color='lightgray', linestyle='-', linewidth=1.5, label='Previsão')
    ax3.axvline(x=TrainLimit, color='orange', linestyle='--', linewidth=2, label=f'Limite Treino (Y = {TrainLimit})')
    
    ax3_sec = ax3.twinx()
    ax3_sec.plot(t_global, np.diff(np.diff(n_prediction, prepend=0), prepend=0), color='green', linestyle='-', linewidth=1.5, label='Data')
    ax3_sec.tick_params(axis='y', labelcolor='green')
    ax3_sec.grid(False)

    
    ax3.set_title(f'EUR/USD - TimeFrame:  | Janela #{TrainLimit}')
    ax3.set_ylabel(f'Preço ')
    ax3.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    plt.close(fig)
    print(f'Gráfico com 3 painéis salvo em: {filename}')








'''
    fig = plt.figure(figsize=(16, 12))
    
    # Configura o GridSpec com 4 linhas e 4 colunas no total:
    # - Linha 0: Ocupa todas as 4 colunas (Gráfico completo)
    # - Linhas 1, 2 e 3: Divididas em 4 colunas cada (Totalizando 12 janelas)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    
    print(f"\n Coluna: {Coluna}")
    # --- 1. Gráfico Principal (Série Completa) ---
    ax1 = fig.add_subplot(gs[0, :])  # Ocupa a linha 0 inteira (todas as colunas)
    ax1.plot(Index_x, Coluna, color='#1f77b4', linewidth=2, label='Série Completa')
    ax1_sec = ax1.twinx()
    ax1_sec.plot(Index_x, np.cumsum(Coluna), color='green', linestyle='-', linewidth=1.5, label='acc')
    ax1.set_title('Visão Geral: Série Completa (Accumulated)', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Acumulado', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # --- 2. As 12 Janelas (3 linhas x 4 colunas = 12 subgráficos) ---
    total_janelas = 12
    
    for i in range(total_janelas):
        inicio = i * tamanho_janela
        fim = inicio + tamanho_janela
        
        # Mapeia o índice para a grade (linhas de 1 a 3, colunas de 0 a 3)
        linha_idx = 1 + (i // 4)
        col_idx = i % 4
        
        ax = fig.add_subplot(gs[linha_idx, col_idx])
        
        # Seleciona o pedaço (slice) do DataFrame
        #dados_janela = df.iloc[inicio:fim]
        x_janela = Index_x[inicio:fim]
        y_janela = Coluna[inicio:fim]
        
        #if not dados_janela.empty:
        if len(x_janela) > 0:
            ax.plot(x_janela, y_janela, color='#ff7f0e', marker='o', markersize=3, linestyle='-')
            ax_sec = ax.twinx()
            ax_sec.plot(x_janela, np.cumsum(y_janela), color='green', linestyle='-', linewidth=1.5, label='acc')
            #-----------------------------------------------------------------------------
            p0 = [y_janela[0], y_janela[0]*1.4, 0.01, 1.0, 0.5] # A,K,B,1,nu
            bounds = ([0.0, 0.0, 0.0, 0.0, 0.0], [y_janela[len(y_janela)-1] * 5, y_janela[len(y_janela)-1], 10.0, 1000.0, 1000.0])
            maxfev=90000000

            try:
                poptN, _ = curve_fit(Richards, x_janela, y_janela, p0=p0, bounds=bounds, maxfev=maxfev)
                #-------------------------------------------------
                Normal_y_predicted = Richards(np.arange(1, len(y_janela) + 1), *poptN)
                sse_acc, rmse_acc, r2_acc = Adjustment_Metrics(y_janela, Normal_y_predicted)
                fit_success = True
            except Exception as e:
                    fit_success = False
                    print(f"\033[33m Erro no ajuste da Curva de Richards na janela: {e}\033[0m")            
            print("-----------------------")
            print(x_janela)
            print("-----------------------+++")
            print(Normal_y_predicted)
            if(fit_success):
                ax_sec.plot(np.arange(1, len(Normal_y_predicted) + 1), Normal_y_predicted, color='red', linestyle='-', linewidth=1.5, label='sim') 
                #ax_sec.plot(x_janela, Normal_y_predicted, color='red', linestyle='-', linewidth=1.5, label='sim')            
            #-----------------------------------------------------------------------------
            ax.set_title(f'Pontos: {inicio} a {fim-1}', fontsize=10, fontweight='semibold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
        else:
            # Caso o DataFrame tenha menos que 240 pontos, desliga os eixos vazios
            ax.axis('off')
            
    # Ajusta o layout para evitar sobreposições
    plt.suptitle('Análise Detalhada por Janelas de Pontos', fontsize=16, fontweight='bold', y=0.95)
    plt.show()
'''