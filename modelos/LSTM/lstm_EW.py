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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from bayes_opt import BayesianOptimization

import math
import os
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
    f.write("model,units,epochs,bs,dropout,rmse,mae,mape,r2,split\n")

# CSV com ytrue e yhat
output_dir2 = "output_ytrue_yhat"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

values_file = os.path.join(output_dir2, f"results_lstm-{timestamp}.csv")

with open(values_file, "w") as f:
    f.write("model,ytrue,yhat\n")  # Cabeçalho do CSV

# Dir com LSTM análises
output_dir3 = "LSTM_analysis"
if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)

metrics_file2 = os.path.join(output_dir3, f"all_metrics-lstm_{timestamp}.csv")
with open(metrics_file2, "w") as f:
    f.write(
        "model,units,epochs,bs,dropout,rmse,rmse_std,mae,mae_std,mape,mape_std,r2,split\n"
    )

results_csv_dir = os.path.join(output_dir3, "results_csv")
if not os.path.exists(results_csv_dir):
    os.makedirs(results_csv_dir)


metrics_list = []

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- LSTM EW VALIDATION --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# Definindo o intervalo de busca para os hiperparâmetros
# Buscas iniciais
# batch_size_options = [16, 32, 64, 128]
# pbounds = {
#     "units": (50, 150),
#     "epochs": (1000, 1000),  # Definindo a faixa para units
#     "batch_size": (0, len(batch_size_options) - 1),  # Faixa para batch  size
#     "dropout": (0.0, 0.5),  # Faixa para dropout
# }

# Busca refinada
batch_size_options = [32, 64]
units_options = [75, 101, 104, 148, 150]

pbounds = {
    "units": (0, len(units_options) - 1),
    "epochs": (1000, 1000),
    "batch_size": (0, len(batch_size_options) - 1),
    "dropout": (0, 0.3),
}

metrics_params_EW = dict(
    best_rmse=float("inf"),
    best_mae=[],
    best_mape=[],
    best_r2=[],
    best_epochs=[],
    best_unit=[],
    best_batch=[],
    best_dropout=[],
)


# Função objetivo que será usada na otimização bayesiana
def treinar_modelo(units, epochs, batch_size, dropout):
    units = units_options[
        int(units)
    ]  # O bayes opt retorna floats, é necessário converter para int
    epochs = int(epochs)
    batch_size = batch_size_options[int(batch_size)]

    # Inicializando as variáveis de resultados
    resultsEW_val = dict(ytrue=[], yhat=[])
    scoringEW_val = dict(rmse=[], mae=[], mape=[], r2=[])

    expanding_window_val = ExpandingWindow(
        n_samples=len(df.loc["2016-01-01":"2018-12-31"]),
        trainw=len(df.loc["2016-01-01":"2017-12-31"]),
        testw=7,
    )

    # Loop para Expanding Window
    for i, (trainidxs, testidxs) in enumerate(expanding_window_val.split(df)):
        X = X_train[trainidxs]
        y = y_train[trainidxs]
        X_t = X_train[testidxs]
        y_t = y_train[testidxs]

        X = X.reshape((X.shape[0], 1, X.shape[1]))
        X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
        y = y.reshape((y.shape[0], 1))

        # Definindo o modelo LSTM
        model = Sequential(
            [
                LSTM(
                    units,
                    dropout=dropout,
                    input_shape=(X.shape[1], X.shape[2]),
                ),
                Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

        # Previsão
        predictions = model.predict(X_t)
        yhat = scaler_target.inverse_transform(predictions)
        ytrue = scaler_target.inverse_transform(y_t)

        resultsEW_val["ytrue"].append(ytrue)
        resultsEW_val["yhat"].append(yhat)

    ytrue_val_EW = []
    yhat_val_EW = []

    # Concatena os resultados
    for i in range(len(resultsEW_val["ytrue"])):
        ytrue_val_EW.extend(resultsEW_val["ytrue"][i])
        yhat_val_EW.extend(resultsEW_val["yhat"][i])

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
    rmse_std = round(np.std(scoringEW_val["rmse"]), 2)
    mae_mean = round(np.mean(scoringEW_val["mae"]), 2)
    mae_std = round(np.std(scoringEW_val["mae"]), 2)
    mape_mean = round((np.mean(scoringEW_val["mape"]) * 100), 2)
    mape_std = round((np.std(scoringEW_val["mape"]) * 100), 2)
    r2_mean = round(np.mean(scoringEW_val["r2"]), 2)

    if rmse_mean < metrics_params_EW["best_rmse"]:
        metrics_params_EW["best_rmse"] = rmse_mean
        metrics_params_EW["best_mae"] = mae_mean
        metrics_params_EW["best_mape"] = mape_mean
        metrics_params_EW["best_r2"] = r2_mean
        metrics_params_EW["best_epochs"] = epochs
        metrics_params_EW["best_unit"] = units
        metrics_params_EW["best_batch"] = batch_size
        metrics_params_EW["best_dropout"] = dropout

    # Write validation metrics to CSV file
    with open(metrics_file2, "a") as f:
        f.write(
            f"LSTM-EW,{units},{epochs},{batch_size},{dropout},{rmse_mean},{rmse_std},{mae_mean},{mae_std},{mape_mean},{mape_std},{r2_mean},val\n"
        )

    ytrue_yhat_df = pd.DataFrame(
        {"ytrue": ytrue_val.flatten(), "yhat": yhat_val.flatten()}
    )
    ytrue_yhat_df.to_csv(
        os.path.join(
            results_csv_dir,
            f"LSTM-EW_{units}_{epochs}_{batch_size}_{dropout}_{timestamp}.csv",
        ),
        index=False,
    )

    return -rmse_mean  # Minimização


# Otimizador Bayesiano
optimizer = BayesianOptimization(
    f=treinar_modelo,
    pbounds=pbounds,
    random_state=None,
)

# Busca inicial
# Executar a otimização
# optimizer.maximize(
#     init_points=15,  # Número de amostras aleatórias para iniciar
#     n_iter=30,  # Número de iterações da otimização
# )

optimizer.maximize(
    init_points=5,  # Número de amostras aleatórias para iniciar
    n_iter=10,  # Número de iterações da otimização
)

with open(metrics_file, "a") as f:
    f.write(
        "LSTM-EW,{},{},{},{},{},{},{},{},val\n".format(
            metrics_params_EW["best_unit"],
            metrics_params_EW["best_epochs"],
            metrics_params_EW["best_batch"],
            metrics_params_EW["best_dropout"],
            metrics_params_EW["best_rmse"],
            metrics_params_EW["best_mae"],
            metrics_params_EW["best_mape"],
            metrics_params_EW["best_r2"],
        )
    )

# Melhor resultado
print("-" * 20)
print("Best_Params_EW_Val:")
for key, value in metrics_params_EW.items():
    print(f"{key}: {value}")
print(optimizer.max)

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- LSTM EW TEST --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

expanding_window_test = ExpandingWindow(
    n_samples=len(df),
    trainw=len(df.loc["2016-01-01":"2018-12-31"]),
    testw=7,
)

resultsEW_test = dict(ytrue=[], yhat=[])
scoringEW = dict(rmse=[], mae=[], mape=[], r2=[])

for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split(df)):
    X = scaled_features[trainidxs]
    y = scaled_target[trainidxs]

    # Dados de teste
    X_t = scaled_features[testidxs]
    y_t = scaled_target[testidxs]

    X = X.reshape((X.shape[0], 1, X.shape[1]))
    X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
    y = y.reshape((y.shape[0], 1))

    # Definição do modelo LSTM
    model = Sequential(
        [
            LSTM(
                metrics_params_EW["best_unit"],
                dropout=metrics_params_EW["best_dropout"],
                input_shape=(X.shape[1], X.shape[2]),
            ),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    # Treinamento do modelo
    model.fit(
        X,
        y,
        epochs=metrics_params_EW["best_epochs"],
        batch_size=metrics_params_EW["best_batch"],
        verbose=0,
        shuffle=False,
    )

    print(
        "Parametros atuais para teste EW: ",
        metrics_params_EW["best_unit"],
        metrics_params_EW["best_epochs"],
        metrics_params_EW["best_batch"],
        metrics_params_EW["best_dropout"],
    )

    # Previsões para os dados de teste
    predictions = model.predict(X_t)

    yhat = scaler_target.inverse_transform(predictions)
    ytrue = scaler_target.inverse_transform(y_t)

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
        # f"LSTM-EW,{metrics_params_EW['best_unit']},{metrics_params_EW['best_epochs']},{metrics_params_EW['best_batch']},{metrics_params_EW['best_dropout']},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
        f"LSTM-EW,{104},{1000},{32},{0.0},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
    )

# Iterar sobre os resultados das janelas de teste
for i in range(len(resultsEW_test["ytrue"])):
    ytrue = resultsEW_test["ytrue"][i]
    yhat = resultsEW_test["yhat"][i]

    # Escrever os valores reais e previstos em uma linha do arquivo CSV
    with open(values_file, "a") as f:
        for true, pred in zip(ytrue, yhat):
            f.write(f"LSTM-EW,{true},{pred}\n")


print("-" * 20)

print("RMSE_EW_Test:", rmse_mean)
print("MAE_EW_Test:", mae_mean)
print("MAPE_EW_Test:", mape_mean)
print("R2_EW_Test:", r2_mean)


# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- CÓDIGO ANTIGO --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


# print("Best_RMSE_EW_Val : ", best_rmse_EW_val)
# print("Best_R2_EW_Val:", best_r2_EW_val)
# print("Best_MAE_EW_Val:", best_mae_EW_val)
# print("Best_MAPE_EW_Val:", best_mape_EW_val)


# expanding_window_val = ExpandingWindow(
#     n_samples=len(df.loc["2016-01-01":"2018-12-31"]),
#     trainw=len(df.loc["2016-01-01":"2017-12-31"]),
#     testw=7,
# )

# best_rmse_EW_val = float("inf")
# best_params_EW = None
# best_r2_EW_val = None
# best_mae_ew_val = None
# best_mape_ew_val = None

# for units, epochs, batch_size, optimizer, dense, verbose, dropout in param_combinations:
#     resultsEW_val = dict(ytrue=[], yhat=[])
#     scoringEW_val = dict(rmse=[], mae=[], mape=[], r2=[])

#     # Loop para Expanding Window
#     for i, (trainidxs, testidxs) in enumerate(expanding_window_val.split(df)):
#         X = X_train[trainidxs]  # tamanho das janelas aumenta de 7 em 7 (expanding)
#         y = y_train[trainidxs]  # tamanho das janelas aumenta de 7 em 7 (expanding)

#         X_t = X_train[testidxs]  # Features para previsão
#         y_t = y_train[testidxs]  # Dados reais para teste

#         X = X.reshape((X.shape[0], 1, X.shape[1]))
#         X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
#         # X.shape[0]: Número de amostras no conjunto de dados X. É o primeiro eixo do array,
#         # que representa o número de exemplos no conjunto de dados.
#         # --------------------------
#         # 1: Número de "steps" de tempo. Neste caso, 1 indica que cada sequência no conjunto de dados tem apenas um "time step".
#         # Ou seja, cada amostra é uma sequência única e indivisível (dados diários).
#         # --------------------------
#         # X.shape[1]: Número de features em cada amostra. É o segundo eixo do array original, que representa o número de
#         # variáveis de entrada para cada exemplo.

#         # Definição do modelo LSTM
#         model = Sequential(
#             [
#                 LSTM(
#                     units,
#                     activation="relu",
#                     dropout=dropout,
#                     input_shape=(X.shape[1], X.shape[2]),
#                 ),
#                 Dense(dense),
#             ]
#         )
#         # X.shape[1] será 1, que é o número de time steps (no caso, um único time step por amostra)
#         # X.shape[2] será o número de features.
#         # model.add(LSTM(units, activation="relu")),

#         model.compile(optimizer=optimizer, loss="mse")

#         # Normal
#         # Best_RMSE_EW_Val :  2201.51
#         # Best_R2_EW_Val: -2.85
#         # Best_MAE_EW_Val: 1883.87
#         # Best_MAPE_EW_Val: 14.81

#         # Stacked
#         # Best_RMSE_EW_Val :  2218.61
#         # Best_R2_EW_Val: -2.85
#         # Best_MAE_EW_Val: 1889.84
#         # Best_MAPE_EW_Val: 14.93

#         # Stacked relu
#         # Best_RMSE_EW_Val :  2310.87
#         # Best_R2_EW_Val: -3.4
#         # Best_MAE_EW_Val: 1975.11
#         # Best_MAPE_EW_Val: 15.46

#         # Normal relu
#         # Best_RMSE_EW_Val :  2290.32
#         # Best_R2_EW_Val: -3.66
#         # Best_MAE_EW_Val: 1974.82
#         # Best_MAPE_EW_Val: 15.45

#         # Treinamento do modelo
#         model.fit(
#             X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False
#         )

#         print(
#             "Parametros atuais: ", units, epochs, optimizer, batch_size, dense, dropout
#         )

#         # Previsões para os dados de validação
#         predictions = model.predict(X_t)

#         yhat = scaler_target.inverse_transform(predictions)
#         ytrue = scaler_target.inverse_transform(y_t)

#         # print("yhat = ", yhat)

#         resultsEW_val["ytrue"].append(ytrue)
#         resultsEW_val["yhat"].append(yhat)

#     ytrue_val_EW = []
#     yhat_val_EW = []

#     # Percorre os resultados e concatena os valores de cada array
#     for i in range(len(resultsEW_val["ytrue"])):
#         ytrue_val_EW.extend(resultsEW_val["ytrue"][i])
#         yhat_val_EW.extend(resultsEW_val["yhat"][i])

#     # Converte as listas para arrays numpy
#     ytrue_val = np.array(ytrue_val_EW)
#     yhat_val = np.array(yhat_val_EW)

# for i in range(len(resultsEW_val["ytrue"]) - 1):
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
#     scoringEW_val["rmse"].append(rmse)
#     scoringEW_val["mae"].append(mae)
#     scoringEW_val["mape"].append(mape)
#     scoringEW_val["r2"].append(r2)

# rmse_mean = round(np.mean(scoringEW_val["rmse"]), 2)
# mae_mean = round(np.mean(scoringEW_val["mae"]), 2)
# mape_mean = round((np.mean(scoringEW_val["mape"]) * 100), 2)
# r2_mean = round(np.mean(scoringEW_val["r2"]), 2)

#     if rmse_mean < best_rmse_EW_val:
#         best_rmse_EW_val = rmse_mean
#         best_mae_EW_val = mae_mean
#         best_mape_EW_val = mape_mean
#         best_r2_EW_val = r2_mean
#         best_epochs_EW = epochs
#         best_unit_EW = units
#         best_batch_size_EW = batch_size
#         best_optimizer_EW = optimizer
#         best_dense_EW = dense
#         best_verbose_EW = verbose
#         best_dropout_EW = dropout

# # Write validation metrics to CSV file
# with open(metrics_file, "a") as f:
#     f.write(
#         f"LSTM-EW,{best_unit_EW},{best_epochs_EW},{best_batch_size_EW},{best_optimizer_EW},{best_dense_EW},{best_verbose_EW},{best_dropout_EW},{best_rmse_EW_val},{best_mae_EW_val},{best_mape_EW_val},{best_r2_EW_val},val\n"
#     )


# print("-" * 20)
# print(
#     "Best_Params_EW_Val",
#     best_unit_EW,
#     best_epochs_EW,
#     best_batch_size_EW,
#     best_optimizer_EW,
#     # best_dense_EW,
#     # best_verbose_EW,
#     best_dropout_EW,
# )
# print("Best_RMSE_EW_Val : ", best_rmse_EW_val)
# print("Best_R2_EW_Val:", best_r2_EW_val)
# print("Best_MAE_EW_Val:", best_mae_EW_val)
# print("Best_MAPE_EW_Val:", best_mape_EW_val)

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
#             LSTM(
#                 metrics_params_EW["best_unit"],
#                 dropout=metrics_params_EW["best_dropout"],
#                 input_shape=(X.shape[1], X.shape[2]),
#             ),
#             Dense(1),
#         ]
#     )
#     model.compile(optimizer="adam", loss="mse")

#     # Treinamento do modelo
#     model.fit(
#         X,
#         y,
#         epochs=metrics_params_EW["best_epochs"],
#         batch_size=metrics_params_EW["best_batch"],
#         verbose=0,
#         shuffle=False,
#     )

#     # print("Len X_train", len(X_train))
#     # # print("X_train", X)
#     # print("Len X_test", len(X_test))
#     # print("X_test", X_test)
#     print(
#         "Parametros atuais para teste EW: ",
#         metrics_params_EW["best_unit"],
#         metrics_params_EW["best_epochs"],
#         metrics_params_EW["best_batch"],
#         metrics_params_EW["best_dropout"],
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
#         f"LSTM-EW,{metrics_params_EW['best_unit']},{metrics_params_EW['best_epochs']},{metrics_params_EW['best_batch']},{metrics_params_EW['best_dropout']},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
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
