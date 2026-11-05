from scipy.optimize import curve_fit
import argparse
import pandas as pd
import numpy as np
import fxutils_v2 as ut

# ==========================================
# CONFIGURAÇÃO DE CORES DA LINHA DE COMANDO
# ==========================================
VERDE = "\033[32m"
VERMELHO = "\033[31m"
AZUL = "\033[34m"
AMARELO = "\033[33m"
RESET = "\033[0m"  # Reseta a cor para o padrão do terminal


# ==========================================
# CONFIGURAÇÃO DE ARGUMENTOS DA LINHA DE COMANDO
# ==========================================
parser = argparse.ArgumentParser(description='Experimento Forex com Curva de Richards e Derivada.')
parser.add_argument('--ref_series', type=str, default='bid', choices=['ask', 'bid'], help="Série de referência")
parser.add_argument('--data_type', type=str, default='Close', choices=['Open', 'High', 'Low', 'Close'])
parser.add_argument('--timeframe', type=str, default='1D', choices=['Tick-By-Tick', '1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W', '1MW'])
parser.add_argument('--data_size', type=int, default=365, help="Conjunto de dados utilizado")
parser.add_argument('--window_size', type=int, default=100, help="Tamanho da Janela (X)")
parser.add_argument('--train_data', type=int, default=50, help="Quantidade de dados de treino (Y)")
#----------------------------------------------------------
args = parser.parse_args()
#----------------------------------------------------------
REF_SERIES      = args.ref_series
DATA_TYPE       = args.data_type
DATA_SIZE       = args.data_size
TIMEFRAME_LABEL = args.timeframe

TRAIN_DATA     = args.train_data
WINDOW         = args.window_size
#----------------------------------------------------------
DB_PATH  = 'forex.duckdb'
db_parametres = {"Path": DB_PATH,
                 "dtserie": REF_SERIES,
                 "dttype": DATA_TYPE,
                 "dtsize": DATA_SIZE,
                 "dtframe": TIMEFRAME_LABEL,
                 "dttrain": TRAIN_DATA,
                 "dtwindow": WINDOW
                }
print(f"{VERDE}Lendo dados de {db_parametres['dtserie'].upper()} para {db_parametres['dtframe']} ({db_parametres['dttype']})...{RESET}")


SystemDF = ut.ReadDataFromDataBase(db_parametres)
#Veriuficar Janelas de Atuação
df = SystemDF[44:69]
#df = SystemDF[54:62]
ut.PlotDataBase(df,args, tamanho_janela=25)


'''
df_limpo = df['Observed'].values
df_limpo = df_limpo - 1
df_limpo = df_limpo * 100000
df_limpo = df_limpo - df_limpo[0] 
#TRAIN_DATA = 5

print("*******************************************")
print(f"Dataframe limpo: {df_limpo}")
print("---------------")
#print(np.cumsum(df['Observed']))
#----------------------------------------------------------
y_data = np.cumsum(df_limpo)
x_data = np.arange(1, len(y_data) + 1)

print("*************")
print(y_data)
print(x_data)
#----------------------------------------------------------
#x_data = np.arange(1, len(y_data) + 1, dtype=float)

#-------------------------------
# Setting the train data vectors
#-------------------------------
StartIndex = 0
FinalIndex = TRAIN_DATA

x_train_data = x_data[StartIndex:FinalIndex+StartIndex]
y_train_data = y_data[StartIndex:FinalIndex+StartIndex]


print("++++++++++++++++++++++++++++++++++++++++++++++")
print(y_train_data)
print(x_train_data)



#-------------------------------
# Preparing richards prediction
#-------------------------------
p0 = [y_train_data[0], y_train_data[0]*1.4, 0.01, 1.0, 0.5] # A,K,B,1,nu
bounds = ([0.0, 0.0, 0.0, 0.0, 0.0], [y_train_data[len(y_train_data)-1] * 5, y_train_data[len(y_train_data)-1], 10.0, 1000.0, 1000.0])
maxfev=90000000

try:
    poptN, _ = curve_fit(ut.Richards, x_train_data, y_train_data, p0=p0, bounds=bounds, maxfev=maxfev)
    #-------------------------------------------------
    y_pred_train = ut.Richards(x_train_data, *poptN)
    sse_train, rmse_train, r2_train = ut.Adjustment_Metrics(y_train_data, y_pred_train)
    #-------------------------------------------------    
    Normal_y_predicted = ut.Richards(x_data, *poptN)
    sse_acc, rmse_acc, r2_acc = ut.Adjustment_Metrics(y_data, Normal_y_predicted)
    fit_success = True

    ut.PlotResults(
                        y_data, 
                        WINDOW,
                        TRAIN_DATA, 
                        0, 
                        FinalIndex, 
                        Normal_y_predicted[:WINDOW],
                        r2_train, 
                        r2_acc,  
                        "teste.png"
                      )
    
except Exception as e:
        fit_success = False
        print(f"{AMARELO}Erro no ajuste da Curva de Richards na janela: {e}{RESET}")

#----------------------------------------------------------
if(fit_success):
    print(*poptN)
    print("--------------------") 
    print(f"R²   Treino:{r2_train:.3f}")
    print(f"R²   Normal:{r2_acc:.3f}")
    #----------------------------------------------------------
    #ut.PlotSimulation(y_data,Normal_y_predicted)
    
'''    
    
'''Parâmetros otimizados: [-1.16707510e+03  6.01830746e+04  2.97168091e-01  5.00000000e+01
  7.98350615e-01]'''
    
    #----------------------------------------------------------
    