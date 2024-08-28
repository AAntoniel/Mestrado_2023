import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import load_model

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

# Separa as features e o target
target = df.iloc[:, 0].T
features = df.iloc[:, 1:]

# Normaliza as features e o target separadamente
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(features)

scaler_target = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

# # X_train - features de 2016 até 2018
# # X_test - features de 2019
# # y_train - dados históricos de 2016 até 2018
# # y_test - dados históricos de 2019
# Divide em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, scaled_target, test_size=0.25, shuffle=False
)

# Imprime as formas dos arrays
# print(train_X.shape, y_train.shape, test_X.shape, y_test.shape)

# Parâmetros do modelo
# units_values = [50, 75, 100]
units_values = [10]
# epochs_values = [1000]
epochs_values = [10]
# batch_size_values = [32, 64, 128]
batch_size_values = [32]
optimizer_values = ["adam"]
dense_values = [1]
verbose_values = [0]
# dropout_values = [0, 0.2, 0.4, 0.6]
dropout_values = [0]

param_combinations = list(
    itertools.product(
        units_values,
        epochs_values,
        batch_size_values,
        optimizer_values,
        dense_values,
        verbose_values,
        dropout_values,
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

metrics_file = os.path.join(output_dir, f"metrics-LSTM_{timestamp}.csv")

with open(metrics_file, "w") as f:
    f.write(
        "model,units,epochs,bs,optimizer,dense,verbose,dropout,rmse,mae,mape,r2,split\n"
    )

# CSV com ytrue e yhat
output_dir2 = "output_ytrue_yhat"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

values_file = os.path.join(output_dir2, f"results_lstm-{timestamp}.csv")

with open(values_file, "w") as f:
    f.write("model,ytrue,yhat\n")  # Cabeçalho do CSV

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- LSTM EW VALIDATION --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

expanding_window_val = ExpandingWindow(
    n_samples=len(df.loc["2016-01-01":"2018-12-31"]),
    trainw=len(df.loc["2016-01-01":"2017-12-31"]),
    testw=7,
)

best_rmse_EW_val = float("inf")
best_params_EW = None
best_r2_EW_val = None
best_mae_ew_val = None
best_mape_ew_val = None

for units, epochs, batch_size, optimizer, dense, verbose, dropout in param_combinations:
    resultsEW_val = dict(ytrue=[], yhat=[])
    scoringEW_val = dict(rmse=[], mae=[], mape=[], r2=[])

    # Loop para Expanding Window
    for i, (trainidxs, testidxs) in enumerate(expanding_window_val.split(df)):
        X = X_train[trainidxs]  # tamanho das janelas aumenta de 7 em 7 (expanding)
        y = y_train[trainidxs]  # tamanho das janelas aumenta de 7 em 7 (expanding)

        X_t = X_train[testidxs]  # Features para previsão
        y_t = y_train[testidxs]  # Dados reais para teste

        X = X.reshape((X.shape[0], 1, X.shape[1]))
        X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
        # X.shape[0]: Número de amostras ou sequências no conjunto de dados X. É o primeiro eixo do array,
        # que representa o número de exemplos no conjunto de dados.
        # --------------------------
        # 1: Número de "steps" de tempo. Neste caso, 1 indica que cada sequência no conjunto de dados tem apenas um "time step".
        # Ou seja, cada amostra é uma sequência única e indivisível (dados diários).
        # --------------------------
        # X.shape[1]: Número de features em cada amostra. É o segundo eixo do array original, que representa o número de
        # variáveis de entrada para cada exemplo.

        # Definição do modelo LSTM
        model = Sequential(
            [
                LSTM(units, input_shape=(X.shape[1], X.shape[2])),
                # X.shape[1] será 1, que é o número de time steps (no caso, um único time step por amostra)
                # X.shape[2] será o número de features.
                Dropout(dropout),
                Dense(dense),
            ]
        )
        model.compile(optimizer=optimizer, loss="mse")

        # Treinamento do modelo
        model.fit(
            X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False
        )

        print(
            "Parametros atuais: ", units, epochs, optimizer, batch_size, dense, dropout
        )

        # Previsões para os dados de validação
        predictions = model.predict(X_t)

        yhat = scaler_target.inverse_transform(predictions)
        ytrue = scaler_target.inverse_transform(y_t)

        print("yhat = ", yhat)

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
        best_epochs_EW = epochs
        best_unit_EW = units
        best_batch_size_EW = batch_size
        best_optimizer_EW = optimizer
        best_dense_EW = dense
        best_verbose_EW = verbose
        best_dropout_EW = dropout

# Write validation metrics to CSV file
with open(metrics_file, "a") as f:
    f.write(
        f"LSTM-EW,{best_unit_EW},{best_epochs_EW},{best_batch_size_EW},{best_optimizer_EW},{best_dense_EW},{best_verbose_EW},{best_dropout_EW},{best_rmse_EW_val},{best_mae_EW_val},{best_mape_EW_val},{best_r2_EW_val},val\n"
    )


print("-" * 20)
print(
    "Best_Params_EW_Val",
    best_unit_EW,
    best_epochs_EW,
    best_batch_size_EW,
    best_optimizer_EW,
    best_dense_EW,
    best_verbose_EW,
    best_dropout_EW,
)
print("Best_RMSE_EW_Val : ", best_rmse_EW_val)
print("Best_R2_EW_Val:", best_r2_EW_val)
print("Best_MAE_EW_Val:", best_mae_EW_val)
print("Best_MAPE_EW_Val:", best_mape_EW_val)

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- LSTM EW TEST --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# expanding_window_test = ExpandingWindow(
#     n_samples=len(df),
#     trainw=len(df.loc["2016-01-01":"2018-12-31"]),
#     testw=7,
# )

# resultsEW_test = dict(ytrue=[], yhat=[])
# scoringEW = dict(rmse=[], mae=[], mape=[], r2=[])

# for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split(df)):
#     X = scaled_features[trainidxs]
#     y = scaled_target[trainidxs]

#     # Dados de teste
#     X_t = scaled_features[testidxs]
#     y_t = scaled_target[testidxs]

#     X = X.reshape((X.shape[0], 1, X.shape[1]))
#     X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))

#     # Definição do modelo LSTM
#     model = Sequential(
#         [
#             LSTM(best_unit_EW, input_shape=(X.shape[1], X.shape[2])),
#             Dropout(best_dropout_EW),
#             Dense(best_dense_EW),
#         ]
#     )
#     model.compile(optimizer=best_optimizer_EW, loss="mse")

#     # Treinamento do modelo
#     model.fit(
#         X,
#         y,
#         epochs=best_epochs_EW,
#         batch_size=best_batch_size_EW,
#         verbose=best_verbose_EW,
#         shuffle=False,
#     )

#     # print("Len X_train", len(X_train))
#     # # print("X_train", X)
#     # print("Len X_test", len(X_test))
#     # print("X_test", X_test)
#     print(
#         "Parametros atuais para teste EW: ",
#         best_unit_EW,
#         best_epochs_EW,
#         best_batch_size_EW,
#         best_dense_EW,
#         best_optimizer_EW,
#         best_dropout_EW,
#     )

#     # Previsões para os dados de teste
#     predictions = model.predict(X_t)

#     yhat = scaler_target.inverse_transform(predictions)
#     ytrue = scaler_target.inverse_transform(y_t)

#     resultsEW_test["ytrue"].append(ytrue)
#     resultsEW_test["yhat"].append(yhat)

# ytrue_test_values = []
# yhat_test_values = []

# # Percorre os resultados e concatena os valores de cada array
# for i in range(len(resultsEW_test["ytrue"])):
#     ytrue_test_values.extend(resultsEW_test["ytrue"][i])
#     yhat_test_values.extend(resultsEW_test["yhat"][i])

# # Converte as listas para arrays numpy
# ytrue_test = np.array(ytrue_test_values)
# yhat_test = np.array(yhat_test_values)

# for i in range(len(resultsEW_test["ytrue"]) - 1):
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
#     scoringEW["rmse"].append(rmse)
#     scoringEW["mae"].append(mae)
#     scoringEW["mape"].append(mape)
#     scoringEW["r2"].append(r2)

# rmse_mean = round(np.mean(scoringEW["rmse"]), 2)
# mae_mean = round(np.mean(scoringEW["mae"]), 2)
# mape_mean = round((np.mean(scoringEW["mape"]) * 100), 2)
# r2_mean = round(np.mean(scoringEW["r2"]), 2)

# # Write test metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"LSTM-EW,{best_unit_EW},{best_epochs_EW},{best_batch_size_EW},{best_optimizer_EW},{best_dense_EW},{best_verbose_EW},{best_dropout_EW},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
#     )

# # Iterar sobre os resultados das janelas de teste
# for i in range(len(resultsEW_test["ytrue"])):
#     ytrue = resultsEW_test["ytrue"][i]
#     yhat = resultsEW_test["yhat"][i]

#     # Escrever os valores reais e previstos em uma linha do arquivo CSV
#     with open(values_file, "a") as f:
#         for true, pred in zip(ytrue, yhat):
#             f.write(f"LSTM-EW,{true},{pred}\n")


# print("-" * 20)

# print("RMSE_EW_Test:", rmse_mean)
# print("MAE_EW_Test:", mae_mean)
# print("MAPE_EW_Test:", mape_mean)
# print("R2_EW_Test:", r2_mean)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LSTM SW2Y VALIDATION ------------------------------------------------
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

# for units, epochs, batch_size, optimizer, dense, verbose, dropout in param_combinations:
#     resultsSW2Y_val = dict(ytrue=[], yhat=[])
#     scoringSW2Y_val = dict(rmse=[], mae=[], mape=[], r2=[])

#     # Loop para Sliding Window
#     for i, (trainidxs, testidxs) in enumerate(SW2Y_val.split(X_train)):
#         X = X_train[trainidxs]
#         y = y_train[trainidxs]

#         # Dados de validação
#         X_t = X_train[testidxs]
#         y_t = y_train[testidxs]

#         X = X.reshape((X.shape[0], 1, X.shape[1]))
#         X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))

#         # Definição do modelo LSTM
#         model = Sequential(
#             [
#                 LSTM(units, input_shape=(X.shape[1], X.shape[2])),
#                 Dropout(dropout),
#                 Dense(dense),
#             ]
#         )
#         model.compile(optimizer=optimizer, loss="mse")

#         # Treinamento do modelo
#         model.fit(
#             X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False
#         )

#         # print("Len X_train", len(X))
#         # # print("X_train", X_train)
#         # print("Len X_test", len(y))
#         # print("X_test", X_test)
#         print(
#             "Parametros atuais de SW2Y: ",
#             units,
#             epochs,
#             batch_size,
#             optimizer,
#             dense,
#             dropout,
#         )

#         # Previsões para os dados de validação
#         predictions = model.predict(X_t)

#         yhat = scaler_target.inverse_transform(predictions)
#         ytrue = scaler_target.inverse_transform(y_t)

#         resultsSW2Y_val["ytrue"].append(ytrue)
#         resultsSW2Y_val["yhat"].append(yhat)

#     ytrue_val_SW2Y = []
#     yhat_val_SW2Y = []

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

#     # print("predictions_2018_EW", predictions_2018_EW)
#     # print("predictions_2018_EW len", len(predictions_2018_EW))
#     # validation_target = validation[:, 0]
#     rmse_mean = round(np.mean(scoringSW2Y_val["rmse"]), 2)
#     mae_mean = round(np.mean(scoringSW2Y_val["mae"]), 2)
#     mape_mean = round((np.mean(scoringSW2Y_val["mape"]) * 100), 2)
#     r2_mean = round(np.mean(scoringSW2Y_val["r2"]), 2)

#     if rmse_mean < best_rmse_SW2Y_val:
#         best_rmse_SW2Y_val = rmse_mean
#         best_mae_SW2Y_val = mae_mean
#         best_mape_SW2Y_val = mape_mean
#         best_r2_SW2Y_val = r2_mean
#         best_unit_SW2Y = units
#         best_epochs_SW2Y = epochs
#         best_batch_size_SW2Y = batch_size
#         best_optimizer_SW2Y = optimizer
#         best_dense_SW2Y = dense
#         best_verbose_SW2Y = verbose
#         best_dropout_SW2Y = dropout

# # Write validation metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"LSTM-SW2Y,{best_unit_SW2Y},{best_epochs_SW2Y},{best_batch_size_SW2Y},{best_optimizer_SW2Y},{best_dense_SW2Y},{best_verbose_SW2Y},{best_dropout_SW2Y},{best_rmse_SW2Y_val},{best_mae_SW2Y_val},{best_mape_SW2Y_val},{best_r2_SW2Y_val},val\n"
#     )

# print("-" * 20)
# print(
#     "Best_Params_SW2Y_Val",
#     best_unit_SW2Y,
#     best_epochs_SW2Y,
#     best_batch_size_SW2Y,
#     best_optimizer_SW2Y,
#     best_dense_SW2Y,
#     best_verbose_SW2Y,
#     best_dropout_SW2Y,
# )
# print("Best_RMSE_SW2Y_Val : ", best_rmse_SW2Y_val)
# print("Best_R2_SW2Y_Val:", best_r2_SW2Y_val)
# print("Best_MAE_SW2Y_Val:", best_mae_SW2Y_val)
# print("Best_MAPE_SW2Y_Val:", best_mape_SW2Y_val)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LSTM SW2Y TEST ------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

# SW2Y_test = SlidingWindow(
#     n_samples=len(df.loc["2017-01-01":"2019-12-31"]),
#     trainw=len(df.loc["2017-01-01":"2018-12-31"]),
#     testw=7,
# )

# resultsSW2Y_test = dict(ytrue=[], yhat=[])
# scoringSW2Y = dict(rmse=[], mae=[], mape=[], r2=[])

# for i, (trainidxs, testidxs) in enumerate(SW2Y_test.split(df)):
#     X = scaled_features[365:][trainidxs]
#     y = scaled_target[365:][trainidxs]

#     # Dados de teste
#     X_t = scaled_features[365:][testidxs]
#     y_t = scaled_target[365:][testidxs]

#     X = X.reshape((X.shape[0], 1, X.shape[1]))
#     X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))

#     # Definição do modelo LSTM
#     model = Sequential(
#         [
#             LSTM(best_unit_SW2Y, input_shape=(X.shape[1], X.shape[2])),
#             Dropout(best_dropout_SW2Y),
#             Dense(best_dense_SW2Y),
#         ]
#     )
#     model.compile(optimizer=best_optimizer_SW2Y, loss="mse")

#     # Treinamento do modelo
#     model.fit(
#         X,
#         y,
#         epochs=best_epochs_SW2Y,
#         batch_size=best_batch_size_SW2Y,
#         verbose=best_verbose_SW2Y,
#         shuffle=False,
#     )

#     print(
#         "Parametros atuais para teste SW2Y: ",
#         best_unit_SW2Y,
#         best_epochs_SW2Y,
#         best_batch_size_SW2Y,
#         best_optimizer_SW2Y,
#         best_dense_SW2Y,
#         best_verbose_SW2Y,
#         best_dropout_SW2Y,
#     )

#     # Previsões para os dados de teste
#     predictions = model.predict(X_t)

#     yhat = scaler_target.inverse_transform(predictions)
#     ytrue = scaler_target.inverse_transform(y_t)

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
#     f.write(
#         f"LSTM-SW2Y,{best_unit_SW2Y},{best_epochs_SW2Y},{best_batch_size_SW2Y},{best_optimizer_SW2Y},{best_dense_SW2Y},{best_verbose_SW2Y},{best_dropout_SW2Y},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
#     )

# # Iterar sobre os resultados das janelas de teste
# for i in range(len(resultsSW2Y_test["ytrue"])):
#     ytrue = resultsSW2Y_test["ytrue"][i]
#     yhat = resultsSW2Y_test["yhat"][i]

#     # Escrever os valores reais e previstos em uma linha do arquivo CSV
#     with open(values_file, "a") as f:
#         for true, pred in zip(ytrue, yhat):
#             f.write(f"LSTM-SW2Y,{true},{pred}\n")

# print("-" * 20)

# print("RMSE_SW2Y_Test:", rmse_mean)
# print("MAE_SW2Y_Test:", mae_mean)
# print("MAPE_SW2Y_Test:", mape_mean)
# print("R2_SW2Y_Test:", r2_mean)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LSTM SW1Y VALIDATION ------------------------------------------------
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

# for units, epochs, batch_size, optimizer, dense, verbose, dropout in param_combinations:
#     resultsSW1Y_val = dict(ytrue=[], yhat=[])
#     scoringSW1Y_val = dict(rmse=[], mae=[], mape=[], r2=[])

#     # Loop para Sliding Window
#     for i, (trainidxs, testidxs) in enumerate(SW1Y_val.split(df)):
#         X = X_train[365:][trainidxs]
#         y = y_train[365:][trainidxs]

#         # Dados de validação
#         X_t = X_train[365:][testidxs]
#         y_t = y_train[365:][testidxs]

#         X = X.reshape((X.shape[0], 1, X.shape[1]))
#         X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))

#         # Definição do modelo LSTM
#         model = Sequential(
#             [
#                 LSTM(units, input_shape=(X.shape[1], X.shape[2])),
#                 Dropout(dropout),
#                 Dense(dense),
#             ]
#         )
#         model.compile(optimizer=optimizer, loss="mse")

#         # Treinamento do modelo
#         model.fit(
#             X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False
#         )

#         # print("Len X_train", len(X))
#         # # print("X_train", X_train)
#         # print("Len X_test", len(y))
#         # print("X_test", X_test)
#         print(
#             "Parametros atuais de SW1Y: ",
#             units,
#             epochs,
#             batch_size,
#             optimizer,
#             dense,
#             dropout,
#         )

#         # Previsões para os dados de validação
#         predictions = model.predict(X_t)

#         yhat = scaler_target.inverse_transform(predictions)
#         ytrue = scaler_target.inverse_transform(y_t)

#         resultsSW1Y_val["ytrue"].append(ytrue)
#         resultsSW1Y_val["yhat"].append(yhat)

#     ytrue_val_SW1Y = []
#     yhat_val_SW1Y = []

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

#     # print("predictions_2018_EW", predictions_2018_EW)
#     # print("predictions_2018_EW len", len(predictions_2018_EW))
#     # validation_target = validation[:, 0]
#     rmse_mean = round(np.mean(scoringSW1Y_val["rmse"]), 2)
#     mae_mean = round(np.mean(scoringSW1Y_val["mae"]), 2)
#     mape_mean = round((np.mean(scoringSW1Y_val["mape"]) * 100), 2)
#     r2_mean = round(np.mean(scoringSW1Y_val["r2"]), 2)

#     if rmse_mean < best_rmse_SW1Y_val:
#         best_rmse_SW1Y_val = rmse_mean
#         best_mae_SW1Y_val = mae_mean
#         best_mape_SW1Y_val = mape_mean
#         best_r2_SW1Y_val = r2_mean
#         best_unit_SW1Y = units
#         best_epochs_SW1Y = epochs
#         best_batch_size_SW1Y = batch_size
#         best_optimizer_SW1Y = optimizer
#         best_dense_SW1Y = dense
#         best_verbose_SW1Y = verbose
#         best_dropout_SW1Y = dropout

# # Write validation metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"LSTM-SW1Y,{best_unit_SW1Y},{best_epochs_SW1Y},{best_batch_size_SW1Y},{best_optimizer_SW1Y},{best_dense_SW1Y},{best_verbose_SW1Y},{best_dropout_SW1Y},{best_rmse_SW1Y_val},{best_mae_SW1Y_val},{best_mape_SW1Y_val},{best_r2_SW1Y_val},val\n"
#     )

# print("-" * 20)
# print(
#     "Best_Params_SW1Y_Val",
#     best_unit_SW1Y,
#     best_epochs_SW1Y,
#     best_batch_size_SW1Y,
#     best_optimizer_SW1Y,
#     best_dense_SW1Y,
#     best_verbose_SW1Y,
#     best_dropout_SW1Y,
# )
# print("Best_RMSE_SW1Y_Val : ", best_rmse_SW1Y_val)
# print("Best_R2_SW1Y_Val:", best_r2_SW1Y_val)
# print("Best_MAE_SW1Y_Val:", best_mae_SW1Y_val)
# print("Best_MAPE_SW1Y_Val:", best_mape_SW1Y_val)

# # ----------------------------------------------------------------------------------------------------------------
# # -------------------------------------------- LSTM SW1Y TEST ------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------

# SW1Y_test = SlidingWindow(
#     n_samples=len(df.loc["2018-01-01":"2019-12-31"]),
#     trainw=len(df.loc["2018-01-01":"2018-12-31"]),
#     testw=7,
# )

# resultsSW1Y_test = dict(ytrue=[], yhat=[])
# scoringSW1Y = dict(rmse=[], mae=[], mape=[], r2=[])

# for i, (trainidxs, testidxs) in enumerate(SW1Y_test.split(df)):
#     X = scaled_features[730:][trainidxs]
#     y = scaled_target[730:][trainidxs]

#     # Dados de teste
#     X_t = scaled_features[730:][testidxs]
#     y_t = scaled_target[730:][testidxs]

#     X = X.reshape((X.shape[0], 1, X.shape[1]))
#     X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))

#     # Definição do modelo LSTM
#     model = Sequential(
#         [
#             LSTM(best_unit_SW1Y, input_shape=(X.shape[1], X.shape[2])),
#             Dropout(best_dropout_SW1Y),
#             Dense(best_dense_SW1Y),
#         ]
#     )
#     model.compile(optimizer=best_optimizer_SW1Y, loss="mse")

#     # Treinamento do modelo
#     model.fit(
#         X,
#         y,
#         epochs=best_epochs_SW1Y,
#         batch_size=best_batch_size_SW1Y,
#         verbose=best_verbose_SW1Y,
#         shuffle=False,
#     )

#     print(
#         "Parametros atuais para teste SW1Y: ",
#         best_unit_SW1Y,
#         best_epochs_SW1Y,
#         best_batch_size_SW1Y,
#         best_optimizer_SW1Y,
#         best_dense_SW1Y,
#         best_dropout_SW1Y,
#     )

#     # Previsões para os dados de teste
#     predictions = model.predict(X_t)

#     yhat = scaler_target.inverse_transform(predictions)
#     ytrue = scaler_target.inverse_transform(y_t)

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
#     f.write(
#         f"LSTM-SW1Y,{best_unit_SW1Y},{best_epochs_SW1Y},{best_batch_size_SW1Y},{best_optimizer_SW1Y},{best_dense_SW1Y},{best_verbose_SW1Y},{best_dropout_SW1Y},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
#     )

# # Iterar sobre os resultados das janelas de teste
# for i in range(len(resultsSW1Y_test["ytrue"])):
#     ytrue = resultsSW1Y_test["ytrue"][i]
#     yhat = resultsSW1Y_test["yhat"][i]

#     # Escrever os valores reais e previstos em uma linha do arquivo CSV
#     with open(values_file, "a") as f:
#         for true, pred in zip(ytrue, yhat):
#             f.write(f"LSTM-SW1Y,{true},{pred}\n")

# print("-" * 20)

# print("RMSE_SW1Y_Test:", rmse_mean)
# print("MAE_SW1Y_Test:", mae_mean)
# print("MAPE_SW1Y_Test:", mape_mean)
# print("R2_SW1Y_Test:", r2_mean)


# --------------------
# Best_Params_SW2Y_Val 200 20 32 1
# Best_RMSE_SW2Y_Val :  2121.7115645950275
# Best_R2_SW2Y_Val: 0.6316805460701567
# Best_MAE_SW2Y_Val: 1589.2042177193166
# Best_MAPE_SW2Y_Val: 0.1260106807398723
# --------------------
# RMSE_SW2Y_Test :  2367.647767442117
# R2_SW2Y_Test: 0.5871888756482156
# MAE_SW2Y_Test: 1791.926024240453
# MAPE_SW2Y_Test: 0.14183656790857965
# --------------------
# Best_Params_SW1Y_Val 300 30 32 1
# Best_RMSE_SW1Y_Val :  2241.8750171595943
# Best_R2_SW1Y_Val: 0.5887794916231792
# Best_MAE_SW1Y_Val: 1664.9277825804147
# Best_MAPE_SW1Y_Val: 0.1303451402703845
# --------------------
# RMSE_SW1Y_Test :  2506.6348696170994
# R2_SW1Y_Test: 0.537300149610946
# MAE_SW1Y_Test: 1937.713805196243
# MAPE_SW1Y_Test: 0.15114201066445354
