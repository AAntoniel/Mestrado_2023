import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import logging
import math
import os
import pickle
import itertools
import datetime


class SlidingWindow:
    #   Vai instânciar a classe e vai necessitar da definição do tamanho das amostras
    def __init__(self, n_samples, trainw, testw):
        self.n_samples = n_samples
        self.trainw = trainw
        self.testw = testw

        self.n_splits = math.ceil((self.n_samples - self.trainw) / testw)

        #       Faz uma verificação e aponta um erro caso as condições não sejam atendidas
        assert n_samples != self.trainw
        assert self.testw > 0

    #   Realiza as divisões para os conjuntos
    def split(self, X, y=None, groups=None):
        #       Gera duas sequências de dados, indo de treino ao final, com o passo de teste
        for i, k in enumerate(range(self.trainw, self.n_samples, self.testw)):

            #           Faz a separação dos dados
            trainidxs = slice(k - self.trainw, k)
            testidxs = slice(k, k + self.testw)

            #           ????
            if i + 1 == self.n_splits:
                testidxs = slice(k, self.n_samples)

            yield trainidxs, testidxs

            if i + 1 == self.n_splits:
                break

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


#   Classe similimar a Sliding Window, mudando somente a divisão dos dados
class ExpandingWindow:
    def __init__(self, n_samples, trainw, testw):
        self.n_samples = n_samples
        self.trainw = trainw
        self.testw = testw

        self.n_splits = math.ceil((self.n_samples - self.trainw) / testw)

        assert n_samples != self.trainw
        assert self.testw > 0

    def split(self, X, y=None, groups=None):
        for i, k in enumerate(range(self.trainw, self.n_samples, self.testw)):
            #           k provavelmente é o conjunto de treinamento, 365 ou 730
            trainidxs = slice(0, k)
            testidxs = slice(k, k + self.testw)

            if i + 1 == self.n_splits:
                testidxs = slice(k, self.n_samples)

            yield trainidxs, testidxs

            if i + 1 == self.n_splits:
                break

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


df = pd.read_csv("data/dataset.csv")
df = df.dropna()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

validation = df.water_produced[730:1095]
test_set = df.water_produced[1095:]
train1YSW = df.water_produced[730:]
train2YSW = df.water_produced[365:]

target = df.iloc[:, 0].T
features = df.iloc[:, 1:]

scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
target_normalized = scaler.fit_transform(target.values.reshape(-1, 1))

# X_train - features de 2016 até 2018
# X_test - features de 2019
# y_train - dados históricos de 2016 até 2018
# y_test - dados históricos de 2019
X_train, X_test, y_train, y_test = train_test_split(
    -features_normalized, target_normalized, test_size=0.25, shuffle=False
)

# p_values = [2, 3, 6]
# d_values = [0, 1]
# q_values = [3, 4, 6]
# P_values = [2, 3]
# D_values = [0, 1]
# Q_values = [1, 3, 4]
# s_values = [7]

p_values = [2]
d_values = [0]
q_values = [3]
P_values = [2]
D_values = [0]
Q_values = [1]
s_values = [365]

param_combinations = list(
    itertools.product(
        p_values, d_values, q_values, P_values, D_values, Q_values, s_values
    )
)

# Obter a data e hora atuais
current_time = datetime.datetime.now()

# Formatar a data e hora no formato desejado
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# Criação de um diretório, se não exisistir
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

metrics_file = os.path.join(output_dir, f"metrics-sarima_{timestamp}.csv")

with open(metrics_file, "w") as f:
    f.write("model,order,sorder,rmse,mae,mape,r2,split\n")

# CSV com ytrue e yhat
output_dir2 = "output_ytrue_yhat"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

values_file = os.path.join(output_dir2, f"results_sarima-{timestamp}.csv")

with open(values_file, "w") as f:
    f.write("model,ytrue,yhat\n")  # Cabeçalho do CSV

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- ARIMA EW VALIDATION --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

expanding_window_val = ExpandingWindow(
    n_samples=len(df.loc["2016-01-01":"2018-12-31"]),
    trainw=len(df.loc["2016-01-01":"2017-12-31"]),
    testw=7,
)

best_rmse_EW_val = float("inf")
best_mae_EW_val = None
best_mape_EW_val = None
best_r2_EW_val = None
best_params_EW = None

for p, d, q, P, D, Q, s in param_combinations:
    arima_order = (p, d, q)
    sarima_order = (P, D, Q, s)

    resultsEW_val = dict(ytrue=[], yhat=[])
    scoringEW_val = dict(rmse=[], mae=[], mape=[], r2=[])

    # Loop para Expanding Window
    for i, (trainidxs, testidxs) in enumerate(expanding_window_val.split(df)):
        X = X_train[trainidxs]
        y = y_train[trainidxs]

        # Dados de validação
        X_t = X_train[testidxs]
        y_t = y_train[testidxs]

        print("ordem atual", arima_order, sarima_order)
        # print("X_train", X_train)
        # print("X_test", X_test)

        sarima_model = sm.tsa.SARIMAX(y, order=arima_order, seasonal_order=sarima_order)
        sarima_fit = sarima_model.fit(
            method="lbfgs",
            low_memory=True,
            cov_type="none",
            start_params=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            maxiter=1000,
        )

        # Previsões para os dados de validação
        predictions = sarima_fit.forecast(steps=len(y_t))

        yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
        ytrue = scaler.inverse_transform(y_t)

        resultsEW_val["ytrue"].append(ytrue)
        resultsEW_val["yhat"].append(yhat)

    ytrue_val_EW = []
    yhat_val_EW = []

    # Percorre os resultados e concatena os valores de cada array
    for i in range(len(resultsEW_val["ytrue"])):
        ytrue_val_EW.extend(resultsEW_val["ytrue"][i])
        yhat_val_EW.extend(resultsEW_val["yhat"][i])

    # Converte as listas para arrays numpy
    ytrue_val = np.array(ytrue_val_EW)
    yhat_val = np.array(yhat_val_EW)

    for i in range(len(resultsEW_val["ytrue"]) - 1):
        # Define o número de dias para esta iteração
        if i == 51:
            num_days = 8
        else:
            num_days = 7

        # Calcula os índices para os últimos dias
        start_idx = i * 7
        end_idx = start_idx + num_days

        # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
        ytrue_last = ytrue_val[start_idx:end_idx]
        yhat_last = yhat_val[start_idx:end_idx]

        # Calcula as métricas para os últimos dias
        rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
        mae = mean_absolute_error(ytrue_last, yhat_last)
        mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
        r2 = r2_score(ytrue_last, yhat_last)

        # Adiciona as métricas ao dicionário de pontuação
        scoringEW_val["rmse"].append(rmse)
        scoringEW_val["mae"].append(mae)
        scoringEW_val["mape"].append(mape)
        scoringEW_val["r2"].append(r2)

    rmse_mean = round(np.mean(scoringEW_val["rmse"]), 2)
    mae_mean = round(np.mean(scoringEW_val["mae"]), 2)
    mape_mean = round((np.mean(scoringEW_val["mape"]) * 100), 2)
    r2_mean = round(np.mean(scoringEW_val["r2"]), 2)

    if rmse_mean < best_rmse_EW_val:
        best_rmse_EW_val = rmse_mean
        best_mae_EW_val = mae_mean
        best_mape_EW_val = mape_mean
        best_r2_EW_val = r2_mean
        best_arima_params_EW = arima_order
        best_sarima_params_EW = sarima_order

# Write validation metrics to CSV file
with open(metrics_file, "a") as f:
    f.write(
        f"SARIMA-EW,{best_arima_params_EW},{best_sarima_params_EW},{best_rmse_EW_val},{best_mae_EW_val},{best_mape_EW_val},{best_r2_EW_val},val\n"
    )

print("Best_Params_EW_Val", best_arima_params_EW, best_sarima_params_EW)
print("Best_RMSE_EW_Val : ", best_rmse_EW_val)
print("Best_R2_EW_Val:", best_r2_EW_val)
print("Best_MAE_EW_Val:", best_mae_EW_val)
print("Best_MAPE_EW_Val:", best_mape_EW_val)

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- ARIMA EW TEST --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

expanding_window_test = ExpandingWindow(
    n_samples=len(df),
    trainw=len(df.loc["2016-01-01":"2018-12-31"]),
    testw=7,
)


resultsEW_test = dict(ytrue=[], yhat=[])
scoringEW = dict(rmse=[], mae=[], mape=[], r2=[])

for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split(df)):
    X = features_normalized[trainidxs]
    y = target_normalized[trainidxs]

    # Dados de teste
    X_t = features_normalized[testidxs]
    y_t = target_normalized[testidxs]

    sarima_model = sm.tsa.SARIMAX(
        y, order=best_arima_params_EW, seasonal_order=best_sarima_params_EW
    )
    sarima_fit = sarima_model.fit(
        method="innovations_mle", low_memory=True, cov_type="none"
    )

    print("Parametros atuais para teste: ", best_arima_params_EW, best_sarima_params_EW)

    # Previsões para os dados de teste
    predictions = sarima_fit.forecast(steps=len(y_t))

    yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
    ytrue = scaler.inverse_transform(y_t)

    resultsEW_test["ytrue"].append(ytrue)
    resultsEW_test["yhat"].append(yhat)

ytrue_test_values = []
yhat_test_values = []

# Percorre os resultados e concatena os valores de cada array
for i in range(len(resultsEW_test["ytrue"])):
    ytrue_test_values.extend(resultsEW_test["ytrue"][i])
    yhat_test_values.extend(resultsEW_test["yhat"][i])

# Converte as listas para arrays numpy
ytrue_test = np.array(ytrue_test_values)
yhat_test = np.array(yhat_test_values)

for i in range(len(resultsEW_test["ytrue"]) - 1):
    # Define o número de dias para esta iteração
    if i == 51:
        num_days = 8
    else:
        num_days = 7

    # Calcula os índices para os últimos dias
    start_idx = i * 7
    end_idx = start_idx + num_days

    # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
    ytrue_last = ytrue_test[start_idx:end_idx]
    yhat_last = yhat_test[start_idx:end_idx]

    # Calcula as métricas para os últimos dias
    rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
    mae = mean_absolute_error(ytrue_last, yhat_last)
    mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
    r2 = r2_score(ytrue_last, yhat_last)

    # Adiciona as métricas ao dicionário de pontuação
    scoringEW["rmse"].append(rmse)
    scoringEW["mae"].append(mae)
    scoringEW["mape"].append(mape)
    scoringEW["r2"].append(r2)

rmse_mean = round(np.mean(scoringEW["rmse"]), 2)
mae_mean = round(np.mean(scoringEW["mae"]), 2)
mape_mean = round((np.mean(scoringEW["mape"]) * 100), 2)
r2_mean = round(np.mean(scoringEW["r2"]), 2)

# Write test metrics to CSV file
with open(metrics_file, "a") as f:
    f.write(
        f"SARIMA-EW,{best_arima_params_EW},{best_sarima_params_EW},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
    )

# Iterar sobre os resultados das janelas de teste
for i in range(len(resultsEW_test["ytrue"])):
    ytrue = resultsEW_test["ytrue"][i]
    yhat = resultsEW_test["yhat"][i]

    # Escrever os valores reais e previstos em uma linha do arquivo CSV
    with open(values_file, "a") as f:
        for true, pred in zip(ytrue, yhat):
            f.write(f"SARIMA-EW,{true},{pred}\n")


print("-" * 20)

print("RMSE_EW_Test:", rmse_mean)
print("MAE_EW_Test:", mae_mean)
print("MAPE_EW_Test:", mape_mean)
print("R2_EW_Test:", r2_mean)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- ARIMA SW2Y VALIDATION ------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

# SW2Y_val = SlidingWindow(
#     n_samples=len(df.loc["2016-01-01":"2018-12-31"]),
#     trainw=len(df.loc["2016-01-01":"2017-12-31"]),
#     testw=7,
# )

# best_rmse_SW2Y_val = float("inf")
# best_mae_SW2Y_val = None
# best_mape_SW2Y_val = None
# best_r2_SW2Y_val = None
# best_params_SW2Y = None

# for p, d, q, P, D, Q, s in param_combinations:
#     arima_order = (p, d, q)
#     sarima_order = (P, D, Q, s)

#     resultsSW2Y_val = dict(ytrue=[], yhat=[])
#     scoringSW2Y_val = dict(rmse=[], mae=[], mape=[], r2=[])

#     # Loop para Expanding Window
#     for i, (trainidxs, testidxs) in enumerate(SW2Y_val.split(df)):
#         X = X_train[trainidxs]
#         y = y_train[trainidxs]

#         # Dados de validação
#         X_t = X_train[testidxs]
#         y_t = y_train[testidxs]

#         print("ordem atual", arima_order, sarima_order)

#         sarima_model = sm.tsa.SARIMAX(y, order=arima_order, seasonal_order=sarima_order)
#         sarima_fit = sarima_model.fit()

#         # Previsões para os dados de validação
#         predictions = sarima_fit.forecast(steps=len(y_t))

#         yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
#         ytrue = scaler.inverse_transform(y_t)

#         resultsSW2Y_val["ytrue"].append(ytrue)
#         resultsSW2Y_val["yhat"].append(yhat)

#     ytrue_val_SW2Y = []
#     yhat_val_SW2Y = []

#     # Percorre os resultados e concatena os valores de cada array
#     for i in range(len(resultsSW2Y_val["ytrue"])):
#         ytrue_val_SW2Y.extend(resultsSW2Y_val["ytrue"][i])
#         yhat_val_SW2Y.extend(resultsSW2Y_val["yhat"][i])

#     # Converte as listas para arrays numpy
#     ytrue_val = np.array(ytrue_val_SW2Y)
#     yhat_val = np.array(yhat_val_SW2Y)

#     for i in range(len(resultsSW2Y_val["ytrue"]) - 1):
#         # Define o número de dias para esta iteração
#         if i == 51:
#             num_days = 8
#         else:
#             num_days = 7

#         # Calcula os índices para os últimos dias
#         start_idx = i * 7
#         end_idx = start_idx + num_days

#         # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
#         ytrue_last = ytrue_val[start_idx:end_idx]
#         yhat_last = yhat_val[start_idx:end_idx]

#         # Calcula as métricas para os últimos dias
#         rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
#         mae = mean_absolute_error(ytrue_last, yhat_last)
#         mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
#         r2 = r2_score(ytrue_last, yhat_last)

#         # Adiciona as métricas ao dicionário de pontuação
#         scoringSW2Y_val["rmse"].append(rmse)
#         scoringSW2Y_val["mae"].append(mae)
#         scoringSW2Y_val["mape"].append(mape)
#         scoringSW2Y_val["r2"].append(r2)

#     rmse_mean = round(np.mean(scoringSW2Y_val["rmse"]), 2)
#     mae_mean = round(np.mean(scoringSW2Y_val["mae"]), 2)
#     mape_mean = round((np.mean(scoringSW2Y_val["mape"]) * 100), 2)
#     r2_mean = round(np.mean(scoringSW2Y_val["r2"]), 2)

#     if rmse_mean < best_rmse_SW2Y_val:
#         best_rmse_SW2Y_val = rmse_mean
#         best_mae_SW2Y_val = mae_mean
#         best_mape_SW2Y_val = mape_mean
#         best_r2_SW2Y_val = r2_mean
#         best_arima_params_SW2Y = arima_order
#         best_sarima_params_SW2Y = sarima_order

# # Write validation metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"SARIMA-SW2Y,{best_arima_params_SW2Y},{best_sarima_params_SW2Y},{best_rmse_SW2Y_val},{best_mae_SW2Y_val},{best_mape_SW2Y_val},{best_r2_SW2Y_val},val\n"
#     )

# print("Best_Params_SW2Y_Val", best_arima_params_SW2Y, best_sarima_params_SW2Y)
# print("Best_RMSE_SW2Y_Val : ", best_rmse_SW2Y_val)
# print("Best_MAE_SW2Y_Val:", best_mae_SW2Y_val)
# print("Best_MAPE_SW2Y_Val:", best_mape_SW2Y_val)
# print("Best_R2_SW2Y_Val:", best_r2_SW2Y_val)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- ARIMA SW2Y TEST ------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

# SW2Y_test = SlidingWindow(
#     n_samples=len(df.loc["2017-01-01":"2019-12-31"]),
#     trainw=len(df.loc["2017-01-01":"2018-12-31"]),
#     testw=7,
# )

# resultsSW2Y_test = dict(ytrue=[], yhat=[])
# scoringSW2Y = dict(rmse=[], mae=[], mape=[], r2=[])

# for i, (trainidxs, testidxs) in enumerate(SW2Y_test.split(df)):
#     X = features_normalized[365:][trainidxs]
#     y = target_normalized[365:][trainidxs]

#     # Dados de teste
#     X_t = features_normalized[365:][testidxs]
#     y_t = target_normalized[365:][testidxs]

#     sarima_model = sm.tsa.SARIMAX(
#         y, order=best_arima_params_SW2Y, seasonal_order=best_sarima_params_SW2Y
#     )
#     sarima_fit = sarima_model.fit()

#     print(
#         "Parametros atuais para teste: ",
#         best_arima_params_SW2Y,
#         best_sarima_params_SW2Y,
#     )

#     # Previsões para os dados de teste
#     predictions = sarima_fit.forecast(steps=len(y_t))

#     yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
#     ytrue = scaler.inverse_transform(y_t)

#     resultsSW2Y_test["ytrue"].append(ytrue)
#     resultsSW2Y_test["yhat"].append(yhat)

# ytrue_test_values = []
# yhat_test_values = []

# # Percorre os resultados e concatena os valores de cada array
# for i in range(len(resultsSW2Y_test["ytrue"])):
#     ytrue_test_values.extend(resultsSW2Y_test["ytrue"][i])
#     yhat_test_values.extend(resultsSW2Y_test["yhat"][i])

# # Converte as listas para arrays numpy
# ytrue_test = np.array(ytrue_test_values)
# yhat_test = np.array(yhat_test_values)

# for i in range(len(resultsSW2Y_test["ytrue"]) - 1):
#     # Define o número de dias para esta iteração
#     if i == 51:
#         num_days = 8
#     else:
#         num_days = 7

#     # Calcula os índices para os últimos dias
#     start_idx = i * 7
#     end_idx = start_idx + num_days

#     # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
#     ytrue_last = ytrue_test[start_idx:end_idx]
#     yhat_last = yhat_test[start_idx:end_idx]

#     # Calcula as métricas para os últimos dias
#     rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
#     mae = mean_absolute_error(ytrue_last, yhat_last)
#     mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
#     r2 = r2_score(ytrue_last, yhat_last)

#     # Adiciona as métricas ao dicionário de pontuação
#     scoringSW2Y["rmse"].append(rmse)
#     scoringSW2Y["mae"].append(mae)
#     scoringSW2Y["mape"].append(mape)
#     scoringSW2Y["r2"].append(r2)

# rmse_mean = round(np.mean(scoringSW2Y["rmse"]), 2)
# mae_mean = round(np.mean(scoringSW2Y["mae"]), 2)
# mape_mean = round((np.mean(scoringSW2Y["mape"]) * 100), 2)
# r2_mean = round(np.mean(scoringSW2Y["r2"]), 2)

# # Write test metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"SARIMA-SW2Y,{best_arima_params_SW2Y},{best_sarima_params_SW2Y},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
#     )

# # Iterar sobre os resultados das janelas de teste
# for i in range(len(resultsSW2Y_test["ytrue"])):
#     ytrue = resultsSW2Y_test["ytrue"][i]
#     yhat = resultsSW2Y_test["yhat"][i]

#     # Escrever os valores reais e previstos em uma linha do arquivo CSV
#     with open(values_file, "a") as f:
#         for true, pred in zip(ytrue, yhat):
#             f.write(f"SARIMA-SW2Y,{true},{pred}\n")

# print("-" * 20)

# print("RMSE_SW2Y_Test:", rmse_mean)
# print("MAE_SW2Y_Test:", mae_mean)
# print("MAPE_SW2Y_Test:", mape_mean)
# print("R2_SW2Y_Test:", r2_mean)


# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- ARIMA SW1Y VALIDATION ------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

# SW1Y_val = SlidingWindow(
#     n_samples=len(df.loc["2017-01-01":"2018-12-31"]),
#     trainw=len(df.loc["2017-01-01":"2017-12-31"]),
#     testw=7,
# )

# best_rmse_SW1Y_val = float("inf")
# best_mae_SW1Y_val = None
# best_mape_SW1Y_val = None
# best_r2_SW1Y_val = None
# best_params_SW1Y = None

# for p, d, q, P, D, Q, s in param_combinations:
#     arima_order = (p, d, q)
#     sarima_order = (P, D, Q, s)

#     resultsSW1Y_val = dict(ytrue=[], yhat=[])
#     scoringSW1Y_val = dict(rmse=[], mae=[], mape=[], r2=[])

#     # Loop para Expanding Window
#     for i, (trainidxs, testidxs) in enumerate(SW1Y_val.split(df)):
#         X = X_train[365:][trainidxs]
#         y = y_train[365:][trainidxs]

#         # Dados de validação
#         X_t = X_train[365:][testidxs]
#         y_t = y_train[365:][testidxs]

#         print("ordem atual", arima_order, sarima_order)

#         sarima_model = sm.tsa.SARIMAX(y, order=arima_order, seasonal_order=sarima_order)
#         sarima_fit = sarima_model.fit()

#         # Previsões para os dados de validação
#         predictions = sarima_fit.forecast(steps=len(y_t))

#         yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
#         ytrue = scaler.inverse_transform(y_t)

#         resultsSW1Y_val["ytrue"].append(ytrue)
#         resultsSW1Y_val["yhat"].append(yhat)

#     ytrue_val_SW1Y = []
#     yhat_val_SW1Y = []

#     # Percorre os resultados e concatena os valores de cada array
#     for i in range(len(resultsSW1Y_val["ytrue"])):
#         ytrue_val_SW1Y.extend(resultsSW1Y_val["ytrue"][i])
#         yhat_val_SW1Y.extend(resultsSW1Y_val["yhat"][i])

#     # Converte as listas para arrays numpy
#     ytrue_val = np.array(ytrue_val_SW1Y)
#     yhat_val = np.array(yhat_val_SW1Y)

#     for i in range(len(resultsSW1Y_val["ytrue"]) - 1):
#         # Define o número de dias para esta iteração
#         if i == 51:
#             num_days = 8
#         else:
#             num_days = 7

#         # Calcula os índices para os últimos dias
#         start_idx = i * 7
#         end_idx = start_idx + num_days

#         # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
#         ytrue_last = ytrue_val[start_idx:end_idx]
#         yhat_last = yhat_val[start_idx:end_idx]

#         # Calcula as métricas para os últimos dias
#         rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
#         mae = mean_absolute_error(ytrue_last, yhat_last)
#         mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
#         r2 = r2_score(ytrue_last, yhat_last)

#         # Adiciona as métricas ao dicionário de pontuação
#         scoringSW1Y_val["rmse"].append(rmse)
#         scoringSW1Y_val["mae"].append(mae)
#         scoringSW1Y_val["mape"].append(mape)
#         scoringSW1Y_val["r2"].append(r2)

#     rmse_mean = round(np.mean(scoringSW1Y_val["rmse"]), 2)
#     mae_mean = round(np.mean(scoringSW1Y_val["mae"]), 2)
#     mape_mean = round((np.mean(scoringSW1Y_val["mape"]) * 100), 2)
#     r2_mean = round(np.mean(scoringSW1Y_val["r2"]), 2)

#     if rmse_mean < best_rmse_SW1Y_val:
#         best_rmse_SW1Y_val = rmse_mean
#         best_mae_SW1Y_val = mae_mean
#         best_mape_SW1Y_val = mape_mean
#         best_r2_SW1Y_val = r2_mean
#         best_arima_params_SW1Y = arima_order
#         best_sarima_params_SW1Y = sarima_order

# # Write validation metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"SARIMA-SW1Y,{best_arima_params_SW1Y},{best_sarima_params_SW1Y},{best_rmse_SW1Y_val},{best_mae_SW1Y_val},{best_mape_SW1Y_val},{best_r2_SW1Y_val},val\n"
#     )

# print("Best_Params_SW1Y_Val", best_arima_params_SW1Y, best_sarima_params_SW1Y)
# print("Best_RMSE_SW1Y_Val : ", best_rmse_SW1Y_val)
# print("Best_MAE_SW1Y_Val:", best_mae_SW1Y_val)
# print("Best_MAPE_SW1Y_Val:", best_mape_SW1Y_val)
# print("Best_R2_SW1Y_Val:", best_r2_SW1Y_val)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- ARIMA SW1Y TEST ------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

# SW1Y_test = SlidingWindow(
#     n_samples=len(df.loc["2018-01-01":"2019-12-31"]),
#     trainw=len(df.loc["2018-01-01":"2018-12-31"]),
#     testw=7,
# )

# resultsSW1Y_test = dict(ytrue=[], yhat=[])
# scoringSW1Y = dict(rmse=[], mae=[], mape=[], r2=[])

# for i, (trainidxs, testidxs) in enumerate(SW1Y_test.split(df)):
#     X = features_normalized[730:][trainidxs]
#     y = target_normalized[730:][trainidxs]

#     # Dados de teste
#     X_t = features_normalized[730:][testidxs]
#     y_t = target_normalized[730:][testidxs]

#     sarima_model = sm.tsa.SARIMAX(
#         y, order=best_arima_params_SW1Y, seasonal_order=best_sarima_params_SW1Y
#     )
#     sarima_fit = sarima_model.fit()

#     print(
#         "Parametros atuais para teste: ", best_arima_params_SW1Y, best_arima_params_SW1Y
#     )

#     # Previsões para os dados de teste
#     predictions = sarima_fit.forecast(steps=len(y_t))

#     yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
#     ytrue = scaler.inverse_transform(y_t)

#     resultsSW1Y_test["ytrue"].append(ytrue)
#     resultsSW1Y_test["yhat"].append(yhat)

# ytrue_test_values = []
# yhat_test_values = []

# # Percorre os resultados e concatena os valores de cada array
# for i in range(len(resultsSW1Y_test["ytrue"])):
#     ytrue_test_values.extend(resultsSW1Y_test["ytrue"][i])
#     yhat_test_values.extend(resultsSW1Y_test["yhat"][i])

# # Converte as listas para arrays numpy
# ytrue_test = np.array(ytrue_test_values)
# yhat_test = np.array(yhat_test_values)

# for i in range(len(resultsSW1Y_test["ytrue"]) - 1):
#     # Define o número de dias para esta iteração
#     if i == 51:
#         num_days = 8
#     else:
#         num_days = 7

#     # Calcula os índices para os últimos dias
#     start_idx = i * 7
#     end_idx = start_idx + num_days

#     # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
#     ytrue_last = ytrue_test[start_idx:end_idx]
#     yhat_last = yhat_test[start_idx:end_idx]

#     # Calcula as métricas para os últimos dias
#     rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
#     mae = mean_absolute_error(ytrue_last, yhat_last)
#     mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
#     r2 = r2_score(ytrue_last, yhat_last)

#     # Adiciona as métricas ao dicionário de pontuação
#     scoringSW1Y["rmse"].append(rmse)
#     scoringSW1Y["mae"].append(mae)
#     scoringSW1Y["mape"].append(mape)
#     scoringSW1Y["r2"].append(r2)

# rmse_mean = round(np.mean(scoringSW1Y["rmse"]), 2)
# mae_mean = round(np.mean(scoringSW1Y["mae"]), 2)
# mape_mean = round((np.mean(scoringSW1Y["mape"]) * 100), 2)
# r2_mean = round(np.mean(scoringSW1Y["r2"]), 2)

# # Write test metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"SARIMA-SW1Y,{best_arima_params_SW1Y},{best_sarima_params_SW1Y},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
#     )

# # Iterar sobre os resultados das janelas de teste
# for i in range(len(resultsSW1Y_test["ytrue"])):
#     ytrue = resultsSW1Y_test["ytrue"][i]
#     yhat = resultsSW1Y_test["yhat"][i]

#     # Escrever os valores reais e previstos em uma linha do arquivo CSV
#     with open(values_file, "a") as f:
#         for true, pred in zip(ytrue, yhat):
#             f.write(f"SARIMA-SW1Y,{true},{pred}\n")

# print("-" * 20)

# print("RMSE_SW1Y_Test:", rmse_mean)
# print("MAE_SW1Y_Test:", mae_mean)
# print("MAPE_SW1Y_Test:", mape_mean)
# print("R2_SW1Y_Test:", r2_mean)
