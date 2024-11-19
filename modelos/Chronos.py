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

import math
import os
import itertools
import datetime

import torch
from chronos import ChronosPipeline


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


# Importação, normalização e divisão dos dados
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

# column = target_normalized.ravel()

# df2 = pd.DataFrame({"water_produced": column})
# print(df2)

# X_train - features de 2016 até 2018
# X_test - features de 2019
# y_train - dados históricos de 2016 até 2018
# y_test - dados históricos de 2019
# X_train, X_test, y_train, y_test = train_test_split(
#     features_normalized, target_normalized, test_size=0.25, shuffle=False
# )

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

metrics_file = os.path.join(output_dir, f"metrics-chronos_{timestamp}.csv")

with open(metrics_file, "w") as f:
    f.write("model,rmse,mae,mape,r2,split\n")

# CSV com ytrue e yhat
output_dir2 = "output_ytrue_yhat"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

values_file = os.path.join(output_dir2, f"results_chronos-{timestamp}.csv")

with open(values_file, "w") as f:
    f.write("model,ytrue,yhat\n")  # Cabeçalho do CSV

output_dir3 = "output_chronos_samples"
if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)

others_file = os.path.join(output_dir3, f"chronos_samples{timestamp}.csv")

with open(others_file, "w") as f:
    f.write("model,preds,iteration\n")


pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- EXEMPLO PARA VERIFICAÇÃO --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# df = pd.read_csv(
#     "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
# )

# print("df: ", df)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# forecast shape: [num_series, num_samples, prediction_length]
# forecast = pipeline.predict(
#     context=torch.tensor(df["water_produced"][:730]),
#     prediction_length=7,
#     num_samples=1,
# )

# # print('df["#Passengers"]: ', df["#Passengers"])
# # print(ChronosPipeline.predict.__doc__)

# forecast_index = pd.date_range(start="1/1/2018", end="1/7/2018")
# low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

# print("forecast: ", forecast)
# print("forecast_index: ", forecast_index)
# # print("median: ", median)

# plt.figure(figsize=(8, 4))
# plt.plot(df["water_produced"][:730], color="royalblue", label="historical data")
# plt.plot(forecast_index, median, color="tomato", label="median forecast")
# plt.fill_between(
#     forecast_index,
#     low,
#     high,
#     color="tomato",
#     alpha=0.3,
#     label="80% prediction interval",
# )
# plt.legend()
# plt.grid()
# plt.show()

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- CHRONOS EW TEST --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

expanding_window_test = ExpandingWindow(
    n_samples=len(df),
    trainw=len(df.loc["2016-01-01":"2018-12-31"]),
    testw=7,
)

best_rmse_EW_val = float("inf")
best_mae_EW_val = None
best_mape_EW_val = None
best_r2_EW_val = None
best_params_EW = None

resultsEW_val = dict(ytrue=[], yhat=[])
scoringEW_val = dict(rmse=[], mae=[], mape=[], r2=[])

print("Expanding Window")
# Loop para Expanding Window
for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split(df)):
    X = features_normalized[trainidxs]
    y = target_normalized[trainidxs]

    # Dados de validação
    X_t = features_normalized[testidxs]
    y_t = target_normalized[testidxs]

    column = y.ravel()
    df2 = pd.DataFrame({"water_produced": column})

    predictions = pipeline.predict(
        context=torch.tensor(df2["water_produced"]),
        prediction_length=len(y_t),
        num_samples=10,
    )

    print(predictions.shape)

    low, median, high = np.quantile(predictions[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    others = scaler.inverse_transform(predictions.reshape(-1, 1))
    yhat = scaler.inverse_transform(median.reshape(-1, 1))
    ytrue = scaler.inverse_transform(y_t)

    resultsEW_val["ytrue"].append(ytrue)
    resultsEW_val["yhat"].append(yhat)

    with open(others_file, "a") as f:
        f.write(f"CHRONOS-EW,{others},{i}\n")

    print(i)


# print(resultsEW_val)

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

# Salva as melhores metricas
with open(metrics_file, "a") as f:
    f.write(f"CHRONOS-EW,{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n")

# Iterar sobre os resultados das janelas de teste
for i in range(len(resultsEW_val["ytrue"])):
    ytrue = resultsEW_val["ytrue"][i]
    yhat = resultsEW_val["yhat"][i]

    # Escreve os valores reais e previstos em uma linha do arquivo CSV
    with open(values_file, "a") as f:
        for true, pred in zip(ytrue, yhat):
            f.write(f"CHRONOS-EW,{true},{pred}\n")

# print("Best_Params_EW_Val", best_params_EW)
print("Best_RMSE_EW_Val : ", best_rmse_EW_val)
print("Best_R2_EW_Val:", best_r2_EW_val)
print("Best_MAE_EW_Val:", best_mae_EW_val)
print("Best_MAPE_EW_Val:", best_mape_EW_val)

# ---------------------------------------------------------------------------------------------
# ------------------------------------ CHRONOS SW2Y TEST---------------------------------------
# ---------------------------------------------------------------------------------------------

sw2y_test = SlidingWindow(
    n_samples=len(df.loc["2017-01-01":"2019-12-31"]),
    trainw=len(df.loc["2017-01-01":"2018-12-31"]),
    testw=7,
)

best_rmse_sw2y_val = float("inf")
best_mae_sw2y_val = None
best_mape_sw2y_val = None
best_r2_sw2y_val = None

results_sw2y_val = dict(ytrue=[], yhat=[])
scoring_sw2y_val = dict(rmse=[], mae=[], mape=[], r2=[])

print("Sliding Window 2Y")
# Loop para Expanding Window
for i, (trainidxs, testidxs) in enumerate(sw2y_test.split(df)):
    X = features_normalized[365:][trainidxs]
    y = target_normalized[365:][trainidxs]

    # Dados de validação
    X_t = features_normalized[365:][testidxs]
    y_t = target_normalized[365:][testidxs]

    column = y.ravel()
    df2 = pd.DataFrame({"water_produced": column})

    predictions = pipeline.predict(
        context=torch.tensor(df2["water_produced"]),
        prediction_length=len(y_t),
        num_samples=10,
    )

    print(predictions.shape)

    low, median, high = np.quantile(predictions[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    others = scaler.inverse_transform(predictions.reshape(-1, 1))
    yhat = scaler.inverse_transform(median.reshape(-1, 1))
    ytrue = scaler.inverse_transform(y_t)

    results_sw2y_val["ytrue"].append(ytrue)
    results_sw2y_val["yhat"].append(yhat)

    with open(others_file, "a") as f:
        f.write(f"CHRONOS-SW2Y,{others},{i}\n")

    print(i)


# print(resultsEW_val)

ytrue_val_sw2y = []
yhat_val_sw2y = []

# Percorre os resultados e concatena os valores de cada array
for i in range(len(results_sw2y_val["ytrue"])):
    ytrue_val_sw2y.extend(results_sw2y_val["ytrue"][i])
    yhat_val_sw2y.extend(results_sw2y_val["yhat"][i])

# Converte as listas para arrays numpy
ytrue_sw2y_val = np.array(ytrue_val_sw2y)
yhat_sw2y_val = np.array(yhat_val_sw2y)

for i in range(len(results_sw2y_val["ytrue"]) - 1):
    # Define o número de dias para esta iteração
    if i == 51:
        num_days = 8
    else:
        num_days = 7

    # Calcula os índices para os últimos dias
    start_idx = i * 7
    end_idx = start_idx + num_days

    # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
    ytrue_last = ytrue_sw2y_val[start_idx:end_idx]
    yhat_last = yhat_sw2y_val[start_idx:end_idx]

    # Calcula as métricas para os últimos dias
    rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
    mae = mean_absolute_error(ytrue_last, yhat_last)
    mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
    r2 = r2_score(ytrue_last, yhat_last)

    # Adiciona as métricas ao dicionário de pontuação
    scoring_sw2y_val["rmse"].append(rmse)
    scoring_sw2y_val["mae"].append(mae)
    scoring_sw2y_val["mape"].append(mape)
    scoring_sw2y_val["r2"].append(r2)

rmse_mean = round(np.mean(scoring_sw2y_val["rmse"]), 2)
mae_mean = round(np.mean(scoring_sw2y_val["mae"]), 2)
mape_mean = round((np.mean(scoring_sw2y_val["mape"]) * 100), 2)
r2_mean = round(np.mean(scoring_sw2y_val["r2"]), 2)

if rmse_mean < best_rmse_sw2y_val:
    best_rmse_sw2y_val = rmse_mean
    best_mae_sw2y_val = mae_mean
    best_mape_sw2y_val = mape_mean
    best_r2_sw2y_val = r2_mean

# Salva as melhores metricas
with open(metrics_file, "a") as f:
    f.write(f"CHRONOS-SW2Y,{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n")

# Iterar sobre os resultados das janelas de teste
for i in range(len(results_sw2y_val["ytrue"])):
    ytrue = results_sw2y_val["ytrue"][i]
    yhat = results_sw2y_val["yhat"][i]

    # Escreve os valores reais e previstos em uma linha do arquivo CSV
    with open(values_file, "a") as f:
        for true, pred in zip(ytrue, yhat):
            f.write(f"CHRONOS-SW2Y,{true},{pred}\n")

# print("Best_Params_EW_Val", best_params_EW)
print("Best_RMSE_sw2y_Val : ", best_rmse_sw2y_val)
print("Best_R2_sw2y_Val:", best_r2_sw2y_val)
print("Best_MAE_sw2y_Val:", best_mae_sw2y_val)
print("Best_MAPE_sw2y_Val:", best_mape_sw2y_val)

# plt.plot(ytrue_sw2y_val)
# plt.plot(yhat_sw2y_val)
# plt.show()

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- CHRONOS SW1Y TEST ------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

SW1Y_test = SlidingWindow(
    n_samples=len(df.loc["2018-01-01":"2019-12-31"]),
    trainw=len(df.loc["2018-01-01":"2018-12-31"]),
    testw=7,
)

resultsSW1Y_test = dict(ytrue=[], yhat=[])
scoringSW1Y = dict(rmse=[], mae=[], mape=[], r2=[])

print("Sliding Window 1Y")
for i, (trainidxs, testidxs) in enumerate(SW1Y_test.split(df)):
    X = features_normalized[730:][trainidxs]
    y = target_normalized[730:][trainidxs]

    # Dados de teste
    X_t = features_normalized[730:][testidxs]
    y_t = target_normalized[730:][testidxs]

    column = y.ravel()
    df2 = pd.DataFrame({"water_produced": column})

    predictions = pipeline.predict(
        context=torch.tensor(df2["water_produced"]),
        prediction_length=len(y_t),
        num_samples=10,
    )

    print(predictions.shape)

    low, median, high = np.quantile(predictions[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    # print(median)
    # print(len(median))
    # print(median.shape)
    # print(predictions)
    # print("len(y)", len(y))
    # print("len(y_t)", len(y_t))

    others = scaler.inverse_transform(predictions.reshape(-1, 1))
    yhat = scaler.inverse_transform(median.reshape(-1, 1))
    ytrue = scaler.inverse_transform(y_t)

    resultsSW1Y_test["ytrue"].append(ytrue)
    resultsSW1Y_test["yhat"].append(yhat)

    with open(others_file, "a") as f:
        f.write(f"CHRONOS-SW1Y,{others},{i}\n")

    print(i)

ytrue_test_values = []
yhat_test_values = []

# Percorre os resultados e concatena os valores de cada array
for i in range(len(resultsSW1Y_test["ytrue"])):
    ytrue_test_values.extend(resultsSW1Y_test["ytrue"][i])
    yhat_test_values.extend(resultsSW1Y_test["yhat"][i])

# Converte as listas para arrays numpy
ytrue_test = np.array(ytrue_test_values)
yhat_test = np.array(yhat_test_values)

for i in range(len(resultsSW1Y_test["ytrue"]) - 1):
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
    scoringSW1Y["rmse"].append(rmse)
    scoringSW1Y["mae"].append(mae)
    scoringSW1Y["mape"].append(mape)
    scoringSW1Y["r2"].append(r2)

rmse_mean = round(np.mean(scoringSW1Y["rmse"]), 2)
mae_mean = round(np.mean(scoringSW1Y["mae"]), 2)
mape_mean = round((np.mean(scoringSW1Y["mape"]) * 100), 2)
r2_mean = round(np.mean(scoringSW1Y["r2"]), 2)

# Salva as melhores metricas
with open(metrics_file, "a") as f:
    f.write(f"CHRONOS-SW1Y,{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n")

# Iterar sobre os resultados das janelas de teste
for i in range(len(resultsSW1Y_test["ytrue"])):
    ytrue = resultsSW1Y_test["ytrue"][i]
    yhat = resultsSW1Y_test["yhat"][i]

    # Escreve os valores reais e previstos em uma linha do arquivo CSV
    with open(values_file, "a") as f:
        for true, pred in zip(ytrue, yhat):
            f.write(f"CHRONOS-SW1Y,{true},{pred}\n")

print("-" * 20)

print("RMSE_SW1Y_Test:", rmse_mean)
print("MAE_SW1Y_Test:", mae_mean)
print("MAPE_SW1Y_Test:", mape_mean)
print("R2_SW1Y_Test:", r2_mean)
