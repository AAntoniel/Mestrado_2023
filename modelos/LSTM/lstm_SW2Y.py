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
import gc


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


# class MyCustomCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         gc.collect()
#         tf.keras.backend.clear_session()


df = pd.read_csv("data/dataset.csv", encoding="UTF-8")
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

# Data e hora atuais
current_time = datetime.datetime.now()

# Formatação data e hora
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# diretórios
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
# -------------------------------------------- LSTM SW2Y VALIDATION --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

batch_size_options = [16, 32, 64, 128]
pbounds = {
    "units": (50, 150),
    "epochs": (1000, 1000),
    "batch_size": (0, len(batch_size_options) - 1),
    "dropout": (0.0, 0.9),
}

metrics_params_sw2y = dict(
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
    units = int(units)  # O bayes opt retorna floats, é necessário converter para int
    epochs = int(epochs)
    batch_size = batch_size_options[int(batch_size)]

    # Inicializando as variáveis de resultados
    results_sw2y_val = dict(ytrue=[], yhat=[])
    scoring_sw2y_val = dict(rmse=[], mae=[], mape=[], r2=[])

    SW2Y_val = SlidingWindow(
        n_samples=len(df.loc["2016-01-01":"2018-12-31"]),
        trainw=len(df.loc["2016-01-01":"2017-12-31"]),
        testw=7,
    )

    # Loop para Expanding Window
    for i, (trainidxs, testidxs) in enumerate(SW2Y_val.split(df)):
        tf.keras.backend.clear_session()
        gc.collect()
        X = X_train[trainidxs]
        y = y_train[trainidxs]
        X_t = X_train[testidxs]
        y_t = y_train[testidxs]

        X = X.reshape((X.shape[0], 1, X.shape[1]))
        X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
        y = y.reshape((y.shape[0], 1))

        dropout = 0 if dropout < 0.01 else dropout

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
        model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
        )

        # Previsão
        predictions = model.predict(X_t)
        yhat = scaler_target.inverse_transform(predictions)
        ytrue = scaler_target.inverse_transform(y_t)

        results_sw2y_val["ytrue"].append(ytrue)
        results_sw2y_val["yhat"].append(yhat)

    ytrue_val_sw2y = []
    yhat_val_sw2y = []

    # Concatena os resultados
    for i in range(len(results_sw2y_val["ytrue"])):
        ytrue_val_sw2y.extend(results_sw2y_val["ytrue"][i])
        yhat_val_sw2y.extend(results_sw2y_val["yhat"][i])

    ytrue_val = np.array(ytrue_val_sw2y)
    yhat_val = np.array(yhat_val_sw2y)

    for i in range(len(results_sw2y_val["ytrue"]) - 1):
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

        # Métricas
        rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
        mae = mean_absolute_error(ytrue_last, yhat_last)
        mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
        r2 = r2_score(ytrue_last, yhat_last)

        scoring_sw2y_val["rmse"].append(rmse)
        scoring_sw2y_val["mae"].append(mae)
        scoring_sw2y_val["mape"].append(mape)
        scoring_sw2y_val["r2"].append(r2)

    rmse_mean = round(np.mean(scoring_sw2y_val["rmse"]), 2)
    rmse_std = round(np.std(scoring_sw2y_val["rmse"]), 2)
    mae_mean = round(np.mean(scoring_sw2y_val["mae"]), 2)
    mae_std = round(np.std(scoring_sw2y_val["mae"]), 2)
    mape_mean = round((np.mean(scoring_sw2y_val["mape"]) * 100), 2)
    mape_std = round((np.std(scoring_sw2y_val["mape"]) * 100), 2)
    r2_mean = round(np.mean(scoring_sw2y_val["r2"]), 2)

    if rmse_mean < metrics_params_sw2y["best_rmse"]:
        metrics_params_sw2y["best_rmse"] = rmse_mean
        metrics_params_sw2y["best_mae"] = mae_mean
        metrics_params_sw2y["best_mape"] = mape_mean
        metrics_params_sw2y["best_r2"] = r2_mean
        metrics_params_sw2y["best_epochs"] = epochs
        metrics_params_sw2y["best_unit"] = units
        metrics_params_sw2y["best_batch"] = batch_size
        metrics_params_sw2y["best_dropout"] = dropout

    with open(metrics_file2, "a") as f:
        f.write(
            f"LSTM-SW2Y,{units},{epochs},{batch_size},{dropout},{rmse_mean},{rmse_std},{mae_mean},{mae_std},{mape_mean},{mape_std},{r2_mean},val\n"
        )

    ytrue_yhat_df = pd.DataFrame(
        {"ytrue": ytrue_val.flatten(), "yhat": yhat_val.flatten()}
    )
    ytrue_yhat_df.to_csv(
        os.path.join(
            results_csv_dir,
            f"LSTM-SW2Y_{units}_{epochs}_{batch_size}_{dropout}_{timestamp}.csv",
        ),
        index=False,
    )

    return -rmse_mean  # Minimização


# Otimizador Bayesiano
optimizer = BayesianOptimization(
    f=treinar_modelo,
    pbounds=pbounds,
    random_state=42,
)

# Busca refinada
optimizer.maximize(
    init_points=20,  # Número de amostras aleatórias para iniciar
    n_iter=30,  # Número de iterações da otimização
)

with open(metrics_file, "a") as f:
    f.write(
        "LSTM-SW2Y,{},{},{},{},{},{},{},{},val\n".format(
            metrics_params_sw2y["best_unit"],
            metrics_params_sw2y["best_epochs"],
            metrics_params_sw2y["best_batch"],
            metrics_params_sw2y["best_dropout"],
            metrics_params_sw2y["best_rmse"],
            metrics_params_sw2y["best_mae"],
            metrics_params_sw2y["best_mape"],
            metrics_params_sw2y["best_r2"],
        )
    )

# Melhor resultado
print("-" * 20)
print("Best_Params_SW2Y_Val:")
for key, value in metrics_params_sw2y.items():
    print(f"{key}: {value}")
print(optimizer.max)

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- LSTM SW2Y TEST --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

SW2Y_test = SlidingWindow(
    n_samples=len(df.loc["2017-01-01":"2019-12-31"]),
    trainw=len(df.loc["2017-01-01":"2018-12-31"]),
    testw=7,
)

results_sw2y_test = dict(ytrue=[], yhat=[])
scoring_sw2y_test = dict(rmse=[], mae=[], mape=[], r2=[])

for i, (trainidxs, testidxs) in enumerate(SW2Y_test.split(df)):
    X = scaled_features[365:][trainidxs]
    y = scaled_target[365:][trainidxs]

    # Dados de teste
    X_t = scaled_features[365:][testidxs]
    y_t = scaled_target[365:][testidxs]

    X = X.reshape((X.shape[0], 1, X.shape[1]))
    X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
    y = y.reshape((y.shape[0], 1))

    model = Sequential(
        [
            LSTM(
                metrics_params_sw2y["best_unit"],
                dropout=metrics_params_sw2y["best_dropout"],
                input_shape=(X.shape[1], X.shape[2]),
            ),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X,
        y,
        epochs=metrics_params_sw2y["best_epochs"],
        batch_size=metrics_params_sw2y["best_batch"],
        verbose=0,
        shuffle=False,
    )

    # print(
    #     "Parametros atuais para teste SW2Y: ",
    #     metrics_params_sw2y["best_unit"],
    #     metrics_params_sw2y["best_epochs"],
    #     metrics_params_sw2y["best_batch"],
    #     metrics_params_sw2y["best_dropout"],
    # )

    # Previsões para os dados de teste
    predictions = model.predict(X_t)

    yhat = scaler_target.inverse_transform(predictions)
    ytrue = scaler_target.inverse_transform(y_t)

    results_sw2y_test["ytrue"].append(ytrue)
    results_sw2y_test["yhat"].append(yhat)

ytrue_test_values = []
yhat_test_values = []

# Percorre os resultados e concatena os valores de cada array
for i in range(len(results_sw2y_test["ytrue"])):
    ytrue_test_values.extend(results_sw2y_test["ytrue"][i])
    yhat_test_values.extend(results_sw2y_test["yhat"][i])

# Converte as listas para arrays numpy
ytrue_test = np.array(ytrue_test_values)
yhat_test = np.array(yhat_test_values)

for i in range(len(results_sw2y_test["ytrue"]) - 1):
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

    # Métricas
    rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
    mae = mean_absolute_error(ytrue_last, yhat_last)
    mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
    r2 = r2_score(ytrue_last, yhat_last)

    scoring_sw2y_test["rmse"].append(rmse)
    scoring_sw2y_test["mae"].append(mae)
    scoring_sw2y_test["mape"].append(mape)
    scoring_sw2y_test["r2"].append(r2)

rmse_mean = round(np.mean(scoring_sw2y_test["rmse"]), 2)
mae_mean = round(np.mean(scoring_sw2y_test["mae"]), 2)
mape_mean = round((np.mean(scoring_sw2y_test["mape"]) * 100), 2)
r2_mean = round(np.mean(scoring_sw2y_test["r2"]), 2)

with open(metrics_file, "a") as f:
    f.write(
        f"LSTM-SW2Y,{metrics_params_sw2y['best_unit']},{metrics_params_sw2y['best_epochs']},{metrics_params_sw2y['best_batch']},{metrics_params_sw2y['best_dropout']},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
    )

# Escrever os valores reais e previstos em arquivo CSV
for i in range(len(results_sw2y_test["ytrue"])):
    ytrue = results_sw2y_test["ytrue"][i]
    yhat = results_sw2y_test["yhat"][i]

    with open(values_file, "a") as f:
        for true, pred in zip(ytrue, yhat):
            f.write(f"LSTM-SW2Y,{true},{pred}\n")


print("-" * 20)

print("RMSE_SW2Y_Test:", rmse_mean)
print("MAE_SW2Y_Test:", mae_mean)
print("MAPE_SW2Y_Test:", mape_mean)
print("R2_SW2Y_Test:", r2_mean)
