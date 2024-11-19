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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import logging
import math
import os
import pickle
import itertools
import datetime


class SlidingWindow:
    def __init__(self, n_samples, trainw, testw):
        self.n_samples = n_samples
        self.trainw = trainw
        self.testw = testw

        self.n_splits = math.ceil((self.n_samples - self.trainw) / testw)

        assert n_samples != self.trainw
        assert self.testw > 0

    def split(self, X, y=None, groups=None):
        for i, k in enumerate(range(self.trainw, self.n_samples, self.testw)):
            trainidxs = slice(k - self.trainw, k)
            testidxs = slice(k, k + self.testw)

            if i + 1 == self.n_splits:
                testidxs = slice(k, self.n_samples)

            yield trainidxs, testidxs

            if i + 1 == self.n_splits:
                break

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


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
    features_normalized, target_normalized, test_size=0.25, shuffle=False
)

# Obter a data e hora atuais
current_time = datetime.datetime.now()

# Formatar a data e hora no formato desejado
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# Criação de um diretório para as métricas, se não exisistir
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

metrics_file = os.path.join(output_dir, f"metrics-LR_{timestamp}.csv")

with open(metrics_file, "w") as f:
    f.write("model,rmse,mae,mape,r2,split\n")

# CSV com ytrue e yhat
output_dir2 = "output_ytrue_yhat"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

values_file = os.path.join(output_dir2, f"results_LR-{timestamp}.csv")

with open(values_file, "w") as f:
    f.write("model,ytrue,yhat\n")  # Cabeçalho do CSV

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- LR EW VALIDATION --------------------------------------------------
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

# param_combinations = list(itertools.product())
resultsEW_val = dict(ytrue=[], yhat=[])
scoringEW_val = dict(rmse=[], mae=[], mape=[], r2=[])

# Loop para Expanding Window
for i, (trainidxs, testidxs) in enumerate(expanding_window_val.split(df)):
    X = X_train[trainidxs]
    y = y_train[trainidxs]

    # Dados de validação
    X_t = X_train[testidxs]
    y_t = y_train[testidxs]

    model = LinearRegression(fit_intercept=True)
    model = model.fit(X, y)

    # Previsões para os dados de validação
    predictions = model.predict(X_t)

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

print(ytrue_val)
print(yhat_val)

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

print("-" * 20)

if rmse_mean < best_rmse_EW_val:
    best_rmse_EW_val = rmse_mean
    best_mae_EW_val = mae_mean
    best_mape_EW_val = mape_mean
    best_r2_EW_val = r2_mean

# Write validation metrics to CSV file
with open(metrics_file, "a") as f:
    f.write(
        f"LR-EW,{best_rmse_EW_val},{best_mae_EW_val},{best_mape_EW_val},{best_r2_EW_val},val\n"
    )


print("Best_RMSE_EW_Val : ", best_rmse_EW_val)
print("Best_MAE_EW_Val:", best_mae_EW_val)
print("Best_MAPE_EW_Val:", best_mape_EW_val)
print("Best_R2_EW_Val:", best_r2_EW_val)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LR EW TEST --------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

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

    model = LinearRegression(fit_intercept=True)
    model = model.fit(X, y)

    predictions = model.predict(X_t)

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

with open(metrics_file, "a") as f:
    f.write(f"LR-EW,{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n")

# Iterar sobre os resultados das janelas de teste
for i in range(len(resultsEW_test["ytrue"])):
    ytrue = resultsEW_test["ytrue"][i]
    yhat = resultsEW_test["yhat"][i]

    # Escrever os valores reais e previstos em uma linha do arquivo CSV
    with open(values_file, "a") as f:
        for true, pred in zip(ytrue, yhat):
            f.write(f"LR-EW,{true},{pred}\n")

print("-" * 20)

print("RMSE_EW_Test:", rmse_mean)
print("MAE_EW_Test:", mae_mean)
print("MAPE_EW_Test:", mape_mean)
print("R2_EW_Test:", r2_mean)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LR SW2Y VALIDATION ------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

# SW2Y_val = SlidingWindow(
#     n_samples=len(df.loc["2016-01-01":"2018-12-31"]),
#     trainw=len(df.loc["2016-01-01":"2017-12-31"]),
#     testw=7,
# )

# best_rmse_SW2Y_val = float("inf")
# best_params_SW2Y = None
# best_r2_SW2Y_val = None
# best_mae_SW2Y_val = None
# best_mape_SW2Y_val = None

# # param_combinations = list(itertools.product())
# resultsSW2Y_val = dict(ytrue=[], yhat=[])
# scoringSW2Y_val = dict(rmse=[], mae=[], mape=[], r2=[])

# # Loop para Expanding Window
# for i, (trainidxs, testidxs) in enumerate(SW2Y_val.split(df)):
#     X = X_train[trainidxs]
#     y = y_train[trainidxs]

#     # Dados de validação
#     X_t = X_train[testidxs]
#     y_t = y_train[testidxs]

#     model = LinearRegression(fit_intercept=True)
#     model = model.fit(X, y)

#     # Previsões para os dados de validação
#     predictions = model.predict(X_t)

#     yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
#     ytrue = scaler.inverse_transform(y_t)

#     resultsSW2Y_val["ytrue"].append(ytrue)
#     resultsSW2Y_val["yhat"].append(yhat)

# ytrue_val_SW2Y = []
# yhat_val_SW2Y = []

# # Percorre os resultados e concatena os valores de cada array
# for i in range(len(resultsSW2Y_val["ytrue"])):
#     ytrue_val_SW2Y.extend(resultsSW2Y_val["ytrue"][i])
#     yhat_val_SW2Y.extend(resultsSW2Y_val["yhat"][i])

# # Converte as listas para arrays numpy
# ytrue_val = np.array(ytrue_val_SW2Y)
# yhat_val = np.array(yhat_val_SW2Y)

# for i in range(len(resultsSW2Y_val["ytrue"]) - 1):
#     # Define o número de dias para esta iteração
#     if i == 51:
#         num_days = 8
#     else:
#         num_days = 7

#     # Calcula os índices para os últimos dias
#     start_idx = i * 7
#     end_idx = start_idx + num_days

#     # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
#     ytrue_last = ytrue_val[start_idx:end_idx]
#     yhat_last = yhat_val[start_idx:end_idx]

#     # Calcula as métricas para os últimos dias
#     rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
#     mae = mean_absolute_error(ytrue_last, yhat_last)
#     mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
#     r2 = r2_score(ytrue_last, yhat_last)

#     # Adiciona as métricas ao dicionário de pontuação
#     scoringSW2Y_val["rmse"].append(rmse)
#     scoringSW2Y_val["mae"].append(mae)
#     scoringSW2Y_val["mape"].append(mape)
#     scoringSW2Y_val["r2"].append(r2)

# rmse_mean = round(np.mean(scoringSW2Y_val["rmse"]), 2)
# mae_mean = round(np.mean(scoringSW2Y_val["mae"]), 2)
# mape_mean = round((np.mean(scoringSW2Y_val["mape"]) * 100), 2)
# r2_mean = round(np.mean(scoringSW2Y_val["r2"]), 2)

# if rmse_mean < best_rmse_SW2Y_val:
#     best_rmse_SW2Y_val = rmse_mean
#     best_mae_SW2Y_val = mae_mean
#     best_mape_SW2Y_val = mape_mean
#     best_r2_SW2Y_val = r2_mean

# # Write validation metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"LR-SW2Y,{best_rmse_SW2Y_val},{best_mae_SW2Y_val},{best_mape_SW2Y_val},{best_r2_SW2Y_val},val\n"
#     )

# print("-" * 20)
# print("Best_RMSE_SW2Y_Val : ", best_rmse_SW2Y_val)
# print("Best_MAE_SW2Y_Val:", best_mae_SW2Y_val)
# print("Best_MAPE_SW2Y_Val:", best_mape_SW2Y_val)
# print("Best_R2_SW2Y_Val:", best_r2_SW2Y_val)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LR SW2Y TEST ------------------------------------------------------
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

#     model = LinearRegression(fit_intercept=True)
#     model = model.fit(X, y)

#     predictions = model.predict(X_t)

#     yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
#     ytrue = scaler.inverse_transform(y_t)

#     resultsSW2Y_test["ytrue"].append(ytrue)
#     resultsSW2Y_test["yhat"].append(yhat)

# ytrue_test_SW2Y = []
# yhat_test_SW2Y = []

# # Percorre os resultados e concatena os valores de cada array
# for i in range(len(resultsSW2Y_test["ytrue"])):
#     ytrue_test_SW2Y.extend(resultsSW2Y_test["ytrue"][i])
#     yhat_test_SW2Y.extend(resultsSW2Y_test["yhat"][i])

# # Converte as listas para arrays numpy
# ytrue_test = np.array(ytrue_test_SW2Y)
# yhat_test = np.array(yhat_test_SW2Y)

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
#     f.write(f"LR-SW2Y,{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n")

# # Iterar sobre os resultados das janelas de teste
# for i in range(len(resultsSW2Y_test["ytrue"])):
#     ytrue = resultsSW2Y_test["ytrue"][i]
#     yhat = resultsSW2Y_test["yhat"][i]

#     # Escrever os valores reais e previstos em uma linha do arquivo CSV
#     with open(values_file, "a") as f:
#         for true, pred in zip(ytrue, yhat):
#             f.write(f"LR-SW2Y,{true},{pred}\n")

# print("-" * 20)

# print("RMSE_SW2Y_Test:", rmse_mean)
# print("MAE_SW2Y_Test:", mae_mean)
# print("MAPE_SW2Y_Test:", mape_mean)
# print("R2_SW2Y_Test:", r2_mean)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LR SW1Y VALIDATION ------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

# SW1Y_val = SlidingWindow(
#     n_samples=len(df.loc["2017-01-01":"2018-12-31"]),
#     trainw=len(df.loc["2017-01-01":"2017-12-31"]),
#     testw=7,
# )

# best_rmse_SW1Y_val = float("inf")
# best_params_SW1Y = None
# best_r2_SW1Y_val = None
# best_mae_SW1Y_val = None
# best_mape_SW1Y_val = None

# # param_combinations = list(itertools.product())
# resultsSW1Y_val = dict(ytrue=[], yhat=[])
# scoringSW1Y_val = dict(rmse=[], mae=[], mape=[], r2=[])

# # Loop para Expanding Window
# for i, (trainidxs, testidxs) in enumerate(SW1Y_val.split(df)):
#     X = X_train[365:][trainidxs]
#     y = y_train[365:][trainidxs]

#     # Dados de validação
#     X_t = X_train[365:][testidxs]
#     y_t = y_train[365:][testidxs]

#     model = LinearRegression(fit_intercept=True)
#     model = model.fit(X, y)

#     # Previsões para os dados de validação
#     predictions = model.predict(X_t)

#     yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
#     ytrue = scaler.inverse_transform(y_t)

#     resultsSW1Y_val["ytrue"].append(ytrue)
#     resultsSW1Y_val["yhat"].append(yhat)

# ytrue_val_SW1Y = []
# yhat_val_SW1Y = []

# # Percorre os resultados e concatena os valores de cada array
# for i in range(len(resultsSW1Y_val["ytrue"])):
#     ytrue_val_SW1Y.extend(resultsSW1Y_val["ytrue"][i])
#     yhat_val_SW1Y.extend(resultsSW1Y_val["yhat"][i])

# # Converte as listas para arrays numpy
# ytrue_val = np.array(ytrue_val_SW1Y)
# yhat_val = np.array(yhat_val_SW1Y)

# for i in range(len(resultsSW1Y_val["ytrue"]) - 1):
#     # Define o número de dias para esta iteração
#     if i == 51:
#         num_days = 8
#     else:
#         num_days = 7

#     # Calcula os índices para os últimos dias
#     start_idx = i * 7
#     end_idx = start_idx + num_days

#     # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
#     ytrue_last = ytrue_val[start_idx:end_idx]
#     yhat_last = yhat_val[start_idx:end_idx]

#     # Calcula as métricas para os últimos dias
#     rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
#     mae = mean_absolute_error(ytrue_last, yhat_last)
#     mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
#     r2 = r2_score(ytrue_last, yhat_last)

#     # Adiciona as métricas ao dicionário de pontuação
#     scoringSW1Y_val["rmse"].append(rmse)
#     scoringSW1Y_val["mae"].append(mae)
#     scoringSW1Y_val["mape"].append(mape)
#     scoringSW1Y_val["r2"].append(r2)

# rmse = round(np.mean(scoringSW1Y_val["rmse"]), 2)
# mae = round(np.mean(scoringSW1Y_val["mae"]), 2)
# mape = round((np.mean(scoringSW1Y_val["mape"]) * 100), 2)
# r2 = round(np.mean(scoringSW1Y_val["r2"]), 2)

# if rmse < best_rmse_SW1Y_val:
#     best_rmse_SW1Y_val = rmse
#     best_mae_SW1Y_val = mae
#     best_mape_SW1Y_val = mape
#     best_r2_SW1Y_val = r2

# # Write validation metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"LR-SW1Y,{best_rmse_SW1Y_val},{best_mae_SW1Y_val},{best_mape_SW1Y_val},{best_r2_SW1Y_val},val\n"
#     )

# print("-" * 20)
# print("Best_RMSE_SW1Y_Val : ", best_rmse_SW1Y_val)
# print("Best_MAE_SW1Y_Val:", best_mae_SW1Y_val)
# print("Best_MAPE_SW1Y_Val:", best_mape_SW1Y_val)
# print("Best_R2_SW1Y_Val:", best_r2_SW1Y_val)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LR SW1Y TEST ------------------------------------------------------
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

#     model = LinearRegression(fit_intercept=True)
#     model = model.fit(X, y)

#     predictions = model.predict(X_t)

#     yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
#     ytrue = scaler.inverse_transform(y_t)

#     resultsSW1Y_test["ytrue"].append(ytrue)
#     resultsSW1Y_test["yhat"].append(yhat)

# ytrue_test_SW1Y = []
# yhat_test_SW1Y = []

# # Percorre os resultados e concatena os valores de cada array
# for i in range(len(resultsSW1Y_test["ytrue"])):
#     ytrue_test_SW1Y.extend(resultsSW1Y_test["ytrue"][i])
#     yhat_test_SW1Y.extend(resultsSW1Y_test["yhat"][i])

# # Converte as listas para arrays numpy
# ytrue_test = np.array(ytrue_test_SW1Y)
# yhat_test = np.array(yhat_test_SW1Y)

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
#     f.write(f"LR-SW1Y,{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n")

# # Iterar sobre os resultados das janelas de teste
# for i in range(len(resultsSW1Y_test["ytrue"])):
#     ytrue = resultsSW1Y_test["ytrue"][i]
#     yhat = resultsSW1Y_test["yhat"][i]

#     # Escrever os valores reais e previstos em uma linha do arquivo CSV
#     with open(values_file, "a") as f:
#         for true, pred in zip(ytrue, yhat):
#             f.write(f"LR-SW1Y,{true},{pred}\n")

# print("-" * 20)

# print("RMSE_SW1Y_Test:", rmse)
# print("MAE_SW1Y_Test:", mae)
# print("MAPE_SW1Y_Test:", mape)
# print("R2_SW1Y_Test:", r2)
