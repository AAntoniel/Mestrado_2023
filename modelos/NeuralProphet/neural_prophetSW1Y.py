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
from neuralprophet import NeuralProphet

from bayes_opt import BayesianOptimization


import math
import os
import datetime

import gc


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

# Data e hora atuais
current_time = datetime.datetime.now()

# Formatação data e hora
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# diretórios
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

metrics_file = os.path.join(output_dir, f"metrics-NP_{timestamp}.csv")

with open(metrics_file, "w") as f:
    f.write("model,n_lags,epochs,bs,lr,rmse,mae,mape,r2,split\n")
    # f.write("model,n_lags,bs,lr,rmse,mae,mape,r2,split\n")

# CSV com ytrue e yhat
output_dir2 = "output_ytrue_yhat"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

values_file = os.path.join(output_dir2, f"results_NP-{timestamp}.csv")

with open(values_file, "w") as f:
    f.write("model,ytrue,yhat\n")  # Cabeçalho do CSV

# Dir com NP análises
output_dir3 = "NP_analysis"
if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)

metrics_file2 = os.path.join(output_dir3, f"all_metrics-NP_{timestamp}.csv")
with open(metrics_file2, "w") as f:
    f.write(
        "model,n_lags,epochs,bs,lr,rmse,rmse_std,mae,mae_std,mape,mape_std,r2,split\n"
        # "model,n_lags,bs,lr,rmse,rmse_std,mae,mae_std,mape,mape_std,r2,split\n"
    )

results_csv_dir = os.path.join(output_dir3, "results_csv")
if not os.path.exists(results_csv_dir):
    os.makedirs(results_csv_dir)


metrics_list = []


# ---------------------------------------------------------------------- DF FERIADOS ----------------------------------------------------------
holiday = pd.DataFrame()
holiday["event"] = df["is_holiday_ctba_gtba_jve"].loc[
    df["is_holiday_ctba_gtba_jve"] == 1
]
holiday = holiday.reset_index()
holiday = holiday.rename(columns={"timestamp": "ds"})
holiday["ds"] = pd.to_datetime(holiday["ds"])
# print(holiday)


def fix_holiday(date):
    month_day = date.strftime("%m-%d")
    fixed_holidays = {
        "01-01": "ano_novo",
        "03-09": "aniversario_de_joinville",
        "04-21": "tiradentes",
        "04-29": "aniversario_de_guaratuba",
        "05-01": "dia_do_trabalho",
        "09-07": "independencia_do_brasil",
        "09-08": "nossa_sra_da_luz_dos_pinhais",
        "10-12": "nossa_sra_aparecida",
        "10-15": "dia_do_professor",
        "10-28": "dia_do_servidor_publico",
        "11-02": "dia_de_finados",
        "11-15": "proclamacao_da_republica",
        "12-25": "natal",
    }
    return fixed_holidays.get(month_day, 1.0)


holiday["event"] = holiday["ds"].apply(fix_holiday)

others = {
    "2016-02-08": "carnaval",
    "2016-02-09": "carnaval",
    "2016-02-10": "carnaval",
    "2016-03-25": "sexta_feira_santa",
    "2016-05-26": "corpus_christi",
    "2017-02-27": "carnaval",
    "2017-02-28": "carnaval",
    "2017-03-01": "carnaval",
    "2017-04-14": "sexta_feira_santa",
    "2017-06-15": "corpus_christi",
    "2018-02-12": "carnaval",
    "2018-02-13": "carnaval",
    "2018-02-14": "carnaval",
    "2018-03-30": "sexta_feira_santa",
    "2018-05-31": "corpus_christi",
    "2019-03-04": "carnaval",
    "2019-03-05": "carnaval",
    "2019-03-06": "carnaval",
    "2019-04-19": "sexta_feira_santa",
    "2019-06-20": "corpus_christi",
}

holiday["event"] = holiday.apply(
    lambda row: others.get(row["ds"].strftime("%Y-%m-%d"), row["event"]), axis=1
)


# def swap_columns(df, col1, col2):
#     col_list = list(df.columns)
#     x, y = col_list.index(col1), col_list.index(col2)
#     col_list[y], col_list[x] = col_list[x], col_list[y]
#     df = df[col_list]
#     return df


# holiday = swap_columns(holiday, "ds", "event")
# print(holiday.dtypes)
# print(holiday.head())


# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- NPROPHET SW1Y VALIDATION --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


n_lags_options = [2, 3, 8]
batch_size_options = [16, 32, 64, 128]

pbounds = {
    "n_lags": (0, len(batch_size_options) - 1),
    "epochs": (180, 180),
    "batch_size": (0, len(batch_size_options) - 1),
    "learning_rate": (0, 0.9),
}

metrics_params_sw1y = dict(
    best_rmse=float("inf"),
    best_mae=[],
    best_mape=[],
    best_r2=[],
    best_epochs=[],
    best_n_lags=[],
    best_batch_size=[],
    best_learning_rate=[],
)


def treinar_modelo(n_lags, epochs, batch_size, learning_rate):
    # def treinar_modelo(n_lags, batch_size, learning_rate):
    n_lags = n_lags_options[
        int(n_lags)
    ]  # O bayes opt retorna floats, é necessário converter para int
    epochs = int(epochs)
    batch_size = batch_size_options[int(batch_size)]

    results_sw1y_val = dict(ytrue=[], yhat=[])
    scoring_sw1y_val = dict(rmse=[], mae=[], mape=[], r2=[])

    SW1Y_val = SlidingWindow(
        n_samples=len(df.loc["2017-01-01":"2018-12-31"]),
        trainw=len(df.loc["2017-01-01":"2017-12-31"]),
        testw=7,
    )

    # Loop para Expanding Window
    for i, (trainidxs, testidxs) in enumerate(SW1Y_val.split(df)):
        gc.collect()
        X = X_train[365:][trainidxs]
        y = y_train[365:][trainidxs]

        # Dados de validação
        X_t = X_train[365:][testidxs]
        y_t = y_train[365:][testidxs]

        df2 = pd.DataFrame()
        start_date = df.index[0]  # Data inicial da janela de treino
        dates = pd.date_range(start=start_date, periods=len(y), freq="D")
        df2["ds"] = dates

        column = y.ravel()
        df2["y"] = column

        # print('len(df2["y"]) validation', len(df2["y"]))

        learning_rate = 0 if learning_rate < 0.01 else learning_rate
        model = NeuralProphet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            epochs=epochs,
            batch_size=batch_size,
            n_lags=n_lags,
            learning_rate=learning_rate,
            n_forecasts=len(y_t),
        )

        model = model.add_events(
            [
                "ano_novo",
                "carnaval",
                "aniversario_de_joinville",
                "sexta_feira_santa",
                "corpus_christi",
                "tiradentes",
                "aniversario_de_guaratuba",
                "dia_do_trabalho",
                "independencia_do_brasil",
                "nossa_sra_da_luz_dos_pinhais",
                "nossa_sra_aparecida",
                "dia_do_professor",
                "dia_do_servidor_publico",
                "dia_de_finados",
                "proclamacao_da_republica",
                "natal",
            ]
        )

        # print(df2)
        history_df = model.create_df_with_events(df2, holiday)
        # model.fit(df2)
        # print(history_df)
        model.fit(history_df, freq="D")

        # future_dates = model.make_future_dataframe(df2, periods=len(y_t))
        future_dates = model.make_future_dataframe(history_df, periods=len(y_t))
        predictions = model.predict(future_dates)

        # Selecionar apenas as colunas de previsão ('yhat1', 'yhat2', ..., 'yhatN')
        yhat_columns = [col for col in predictions.columns if col.startswith("yhat")]

        predicted_values = []

        # Iterar pelas colunas e adicionar os valores não-nulos à lista
        for col in yhat_columns:
            predicted_values.extend(predictions[col].dropna().tolist())

        predicted_values = np.array(predicted_values)

        yhat = scaler.inverse_transform(predicted_values.reshape(-1, 1))
        ytrue = scaler.inverse_transform(y_t)

        results_sw1y_val["ytrue"].append(ytrue)
        results_sw1y_val["yhat"].append(yhat)

    ytrue_val_sw1y = []
    yhat_val_sw1y = []

    # Percorre os resultados e concatena os valores de cada array
    for i in range(len(results_sw1y_val["ytrue"])):
        ytrue_val_sw1y.extend(results_sw1y_val["ytrue"][i])
        yhat_val_sw1y.extend(results_sw1y_val["yhat"][i])

    # Converte as listas para arrays numpy
    ytrue_val = np.array(ytrue_val_sw1y)
    yhat_val = np.array(yhat_val_sw1y)

    for i in range(len(results_sw1y_val["ytrue"]) - 1):
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

        # print(ytrue_last)
        # print(yhat_last)

        # Métricas
        rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
        mae = mean_absolute_error(ytrue_last, yhat_last)
        mape = mean_absolute_percentage_error(ytrue_last, yhat_last)
        r2 = r2_score(ytrue_last, yhat_last)

        scoring_sw1y_val["rmse"].append(rmse)
        scoring_sw1y_val["mae"].append(mae)
        scoring_sw1y_val["mape"].append(mape)
        scoring_sw1y_val["r2"].append(r2)

    rmse_mean = round(np.mean(scoring_sw1y_val["rmse"]), 2)
    rmse_std = round(np.std(scoring_sw1y_val["rmse"]), 2)
    mae_mean = round(np.mean(scoring_sw1y_val["mae"]), 2)
    mae_std = round(np.std(scoring_sw1y_val["mae"]), 2)
    mape_mean = round((np.mean(scoring_sw1y_val["mape"]) * 100), 2)
    mape_std = round((np.std(scoring_sw1y_val["mape"]) * 100), 2)
    r2_mean = round(np.mean(scoring_sw1y_val["r2"]), 2)

    # print("-" * 20)

    if rmse_mean < metrics_params_sw1y["best_rmse"]:
        metrics_params_sw1y["best_rmse"] = rmse_mean
        metrics_params_sw1y["best_mae"] = mae_mean
        metrics_params_sw1y["best_mape"] = mape_mean
        metrics_params_sw1y["best_r2"] = r2_mean
        metrics_params_sw1y["best_epochs"] = epochs
        metrics_params_sw1y["best_n_lags"] = n_lags
        metrics_params_sw1y["best_batch_size"] = batch_size
        metrics_params_sw1y["best_learning_rate"] = learning_rate

    with open(metrics_file2, "a") as f:
        f.write(
            f"NP-SW1Y,{n_lags},{epochs},{batch_size},{learning_rate},{rmse_mean},{rmse_std},{mae_mean},{mae_std},{mape_mean},{mape_std},{r2_mean},val\n"
            # f"NP-SW1Y,{n_lags},{batch_size},{learning_rate},{rmse_mean},{rmse_std},{mae_mean},{mae_std},{mape_mean},{mape_std},{r2_mean},val\n"
        )

    ytrue_yhat_df = pd.DataFrame(
        {"ytrue": ytrue_val.flatten(), "yhat": yhat_val.flatten()}
    )
    ytrue_yhat_df.to_csv(
        os.path.join(
            results_csv_dir,
            f"NP-SW1Y_{n_lags}_{epochs}_{batch_size}_{learning_rate}_{timestamp}.csv",
            # f"NP-SW1Y_{n_lags}_{batch_size}_{learning_rate}_{timestamp}.csv",
        ),
        index=False,
    )

    return -rmse_mean


# Otimizador Bayesiano
optimizer = BayesianOptimization(
    f=treinar_modelo,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=20,  # Número de amostras aleatórias para iniciar
    n_iter=30,  # Número de iterações da otimização
)

with open(metrics_file, "a") as f:
    f.write(
        "NP-SW1Y,{},{},{},{},{},{},{},{},val\n".format(
            metrics_params_sw1y["best_n_lags"],
            metrics_params_sw1y["best_epochs"],
            metrics_params_sw1y["best_batch_size"],
            metrics_params_sw1y["best_learning_rate"],
            metrics_params_sw1y["best_rmse"],
            metrics_params_sw1y["best_mae"],
            metrics_params_sw1y["best_mape"],
            metrics_params_sw1y["best_r2"],
        )
    )

# Melhor resultado
print("-" * 20)
print("Best_Params_SW1Y_Val:")
for key, value in metrics_params_sw1y.items():
    print(f"{key}: {value}")
print(optimizer.max)

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------- LSTM SW1Y TEST --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

SW1Y_test = SlidingWindow(
    n_samples=len(df.loc["2018-01-01":"2019-12-31"]),
    trainw=len(df.loc["2018-01-01":"2018-12-31"]),
    testw=7,
)

results_sw1y_test = dict(ytrue=[], yhat=[])
scoring_sw1y_test = dict(rmse=[], mae=[], mape=[], r2=[])

for i, (trainidxs, testidxs) in enumerate(SW1Y_test.split(df)):
    X = features_normalized[730:][trainidxs]
    y = target_normalized[730:][trainidxs]

    # Dados de teste
    X_t = features_normalized[730:][testidxs]
    y_t = target_normalized[730:][testidxs]

    df2 = pd.DataFrame()
    start_date = df.index[0]  # Data inicial da janela de treino
    dates = pd.date_range(start=start_date, periods=len(y), freq="D")
    df2["ds"] = dates

    column = y.ravel()
    df2["y"] = column

    print('len(df2["y"]) teste', len(df2["y"]))

    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=metrics_params_sw1y["best_epochs"],
        batch_size=metrics_params_sw1y["best_batch_size"],
        n_lags=metrics_params_sw1y["best_n_lags"],
        learning_rate=metrics_params_sw1y["best_learning_rate"],
        n_forecasts=len(y_t),
    )

    model = model.add_events(
        [
            "ano_novo",
            "carnaval",
            "aniversario_de_joinville",
            "sexta_feira_santa",
            "corpus_christi",
            "tiradentes",
            "aniversario_de_guaratuba",
            "dia_do_trabalho",
            "independencia_do_brasil",
            "nossa_sra_da_luz_dos_pinhais",
            "nossa_sra_aparecida",
            "dia_do_professor",
            "dia_do_servidor_publico",
            "dia_de_finados",
            "proclamacao_da_republica",
            "natal",
        ]
    )

    print(
        "Parametros atuais para teste SW1Y: ",
        metrics_params_sw1y["best_n_lags"],
        metrics_params_sw1y["best_epochs"],
        metrics_params_sw1y["best_batch_size"],
        metrics_params_sw1y["best_learning_rate"],
    )
    # model.fit(df2)
    history_df = model.create_df_with_events(df2, holiday)
    # model.fit(df2)
    # print(history_df)
    model.fit(history_df, freq="D")

    future_dates = model.make_future_dataframe(history_df, periods=len(y_t))
    # print(future_dates)
    predictions = model.predict(future_dates)

    # Selecionar apenas as colunas de previsão ('yhat1', 'yhat2', ..., 'yhatN')
    yhat_columns = [col for col in predictions.columns if col.startswith("yhat")]

    predicted_values = []

    # Iterar pelas colunas e adicionar os valores não-nulos à lista
    for col in yhat_columns:
        predicted_values.extend(predictions[col].dropna().tolist())

    predicted_values = np.array(predicted_values)

    yhat = scaler.inverse_transform(predicted_values.reshape(-1, 1))
    ytrue = scaler.inverse_transform(y_t)

    results_sw1y_test["ytrue"].append(ytrue)
    results_sw1y_test["yhat"].append(yhat)


ytrue_test_values = []
yhat_test_values = []

# Percorre os resultados e concatena os valores de cada array
for i in range(len(results_sw1y_test["ytrue"])):
    ytrue_test_values.extend(results_sw1y_test["ytrue"][i])
    yhat_test_values.extend(results_sw1y_test["yhat"][i])

# Converte as listas para arrays numpy
ytrue_test = np.array(ytrue_test_values)
yhat_test = np.array(yhat_test_values)

for i in range(len(results_sw1y_test["ytrue"]) - 1):
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

    scoring_sw1y_test["rmse"].append(rmse)
    scoring_sw1y_test["mae"].append(mae)
    scoring_sw1y_test["mape"].append(mape)
    scoring_sw1y_test["r2"].append(r2)

rmse_mean = round(np.mean(scoring_sw1y_test["rmse"]), 2)
mae_mean = round(np.mean(scoring_sw1y_test["mae"]), 2)
mape_mean = round((np.mean(scoring_sw1y_test["mape"]) * 100), 2)
r2_mean = round(np.mean(scoring_sw1y_test["r2"]), 2)

with open(metrics_file, "a") as f:
    f.write(
        f"NP-SW1Y,{metrics_params_sw1y['best_n_lags']},{metrics_params_sw1y['best_epochs']},{metrics_params_sw1y['best_batch_size']},{metrics_params_sw1y['best_learning_rate']},{rmse_mean},{mae_mean},{mape_mean},{r2_mean},test\n"
    )

# Escrever os valores reais e previstos em arquivo CSV
for i in range(len(results_sw1y_test["ytrue"])):
    ytrue = results_sw1y_test["ytrue"][i]
    yhat = results_sw1y_test["yhat"][i]
    
    with open(values_file, "a") as f:
        for true, pred in zip(ytrue, yhat):
            f.write(f"NP-SW1Y,{true},{pred}\n")


print("-" * 20)

print("RMSE_SW1Y_Test:", rmse_mean)
print("MAE_SW1Y_Test:", mae_mean)
print("MAPE_SW1Y_Test:", mape_mean)
print("R2_SW1Y_Test:", r2_mean)
