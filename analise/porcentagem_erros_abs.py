import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import datetime
import os

from plot import setup, wrapup, save

matplotlib.use("TkAgg")
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")


class analise_abs(object):
    def __init__(self, ytrue, yhat):
        self.ytrue = pd.to_numeric(ytrue)
        self.yhat = pd.to_numeric(yhat)

    def biggest_errors(self, perc, output_dir, model_name, start, end):
        self.perc = perc
        if not isinstance(self.ytrue.index, pd.DatetimeIndex) or not isinstance(
            self.yhat.index, pd.DatetimeIndex
        ):
            dates = pd.date_range(start=start, end=end)
            self.ytrue.index = dates
            self.yhat.index = dates

        self.ytrue = pd.Series(self.ytrue)
        self.yhat = pd.Series(self.yhat)

        self.abs_errors = abs((self.ytrue - self.yhat) / self.ytrue) * 100
        # self.abs_errors = abs(self.ytrue - self.yhat)
        self.abs_errors_desc = self.abs_errors.sort_values(ascending=False)
        self.n_errors = int(len(self.abs_errors_desc) * (self.perc / 100))

        holidays_file = os.path.join(output_dir, f"{model_name}_top_errors.csv")
        with open(holidays_file, "w") as f:
            f.write(f"{self.abs_errors_desc[: self.n_errors]}")
        print("Number of errors: ", len(self.abs_errors_desc[: self.n_errors]))
        print(
            f"The top {self.perc}% biggest errors are: ",
            self.abs_errors_desc[: self.n_errors],
        )

    def plot_abs_errors(self, output_dir, model_name):
        plt.figure(figsize=(11.69, 8.27))
        plt.plot(self.abs_errors_desc.values)
        plt.axvspan(0, self.n_errors, color="red", alpha=0.3)
        plt.title(
            f"Top {self.perc} Percentual de Erros ({model_name})",
            fontsize=16,
        )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_top_errors.pdf"), format="pdf"
        )
        plt.clf()

        plt.figure(figsize=(11.69, 8.27))
        plt.plot(self.ytrue, label="Série temporal original")
        plt.plot(self.abs_errors, label="Absolute percentage errors")

        top_errors_dates = self.abs_errors_desc.head(self.n_errors).index
        top_errors_values = self.abs_errors_desc.head(self.n_errors).values

        for date in top_errors_dates:
            plt.axvspan(date, date, color="red", alpha=0.4, lw=1)

        plt.scatter(
            top_errors_dates,
            top_errors_values,
            color="red",
            label="Top Erros",
            zorder=3,
        )

        plt.legend(fontsize=12)
        plt.title(f"Série Original x APE ({model_name})", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_errors_inST.pdf"), format="pdf"
        )
        plt.clf()


output_dir1 = "comparativos/analise_erros_abs_per"
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)

# df_arima = pd.read_csv("output_ytrue_yhat/results_arima-2024-05-20-15-38-45.csv")
# df_arima = pd.DataFrame(df_arima)
# df_arima["ytrue"] = (
#     df_arima["ytrue"].str.replace(r"[\[\]]", "", regex=True).astype(float)
# )
# df_arima["yhat"] = df_arima["yhat"].str.replace(r"[\[\]]", "", regex=True).astype(float)

"""#Divisão dos dados reais e teste para cada modelo"""

df_metrics_arima = pd.read_csv("output/metrics-arima_2024-05-20-15-38-45.csv")

df_metrics_sarima1 = pd.read_csv("output/metrics-sarima_2024-05-22-09-36-40.csv")
df_metrics_sarima2 = pd.read_csv("output/metrics-sarima_2024-05-25-01-55-14.csv")

df_metrics_lstm1 = pd.read_csv("output/metrics-LSTM_2024-06-27-10-33-50.csv")  # EW
df_metrics_lstm2 = pd.read_csv("output/metrics-LSTM_2024-06-28-23-29-24.csv")  # SW2Y
df_metrics_lstm3 = pd.read_csv("output/metrics-LSTM_2025-02-20-09-52-26.csv")  # SW1Y

df_metrics_np1 = pd.read_csv("output/metrics-NP_2024-12-12-20-29-38.csv")  # EW
df_metrics_np2 = pd.read_csv("output/metrics-NP_2024-12-16-20-57-34.csv")  # SW2Y
df_metrics_np3 = pd.read_csv("output/metrics-NP_2024-12-26-14-24-33.csv")  # SW1Y

df_metrics_cr = pd.read_csv("output/metrics-chronos_2024-10-24-13-12-42.csv")  # EW

df_values_arima = pd.read_csv("output_ytrue_yhat/results_arima-2024-05-20-15-38-45.csv")

df_values_sarima1 = pd.read_csv(
    "output_ytrue_yhat/results_sarima-2024-05-22-09-36-40.csv"
)
df_values_sarima2 = pd.read_csv(
    "output_ytrue_yhat/results_sarima-2024-05-25-01-55-14.csv"
)

df_values_lstm1 = pd.read_csv("output_ytrue_yhat/results_lstm-2024-06-27-10-33-50.csv")
df_values_lstm2 = pd.read_csv("output_ytrue_yhat/results_lstm-2024-06-28-23-29-24.csv")
df_values_lstm3 = pd.read_csv("output_ytrue_yhat/results_lstm-2025-02-20-09-52-26.csv")

df_values_np1 = pd.read_csv("output_ytrue_yhat/results_NP-2024-12-12-20-29-38.csv")
df_values_np2 = pd.read_csv("output_ytrue_yhat/results_NP-2024-12-16-20-57-34.csv")
df_values_np3 = pd.read_csv("output_ytrue_yhat/results_NP-2024-12-26-14-24-33.csv")

df_values_cr = pd.read_csv("output_ytrue_yhat/results_chronos-2024-10-24-13-12-42.csv")

dates = pd.date_range(start="2019-01-01", end="2019-12-31")

# ---------------------//----------------------------------//---------------------------------------

# Divisão dos dados de teste para cada backtest de ARIMA
# Expanding Window
test_ew_arima = df_values_arima[:365]
test_ew_arima.index = dates
test_ew_arima.ytrue = test_ew_arima.ytrue.apply(lambda x: float(x.strip("[]")))
test_ew_arima.yhat = test_ew_arima.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 2Y
test_sw2y_arima = df_values_arima[365:730]
test_sw2y_arima.index = dates
test_sw2y_arima.ytrue = test_sw2y_arima.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw2y_arima.yhat = test_sw2y_arima.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 1Y
test_sw1y_arima = df_values_arima[730:]
test_sw1y_arima.index = dates
test_sw1y_arima.ytrue = test_sw1y_arima.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw1y_arima.yhat = test_sw1y_arima.yhat.apply(lambda x: float(x.strip("[]")))

# ---------------------//----------------------------------//---------------------------------------

# Divisão dos dados de teste para cada backtest de SARIMA
# Expanding Window
test_ew_sarima = df_values_sarima1[:365]
test_ew_sarima.index = dates
test_ew_sarima.ytrue = test_ew_sarima.ytrue.apply(lambda x: float(x.strip("[]")))
test_ew_sarima.yhat = test_ew_sarima.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 2Y
test_sw2y_sarima = df_values_sarima1[365:730]
test_sw2y_sarima.index = dates
test_sw2y_sarima.ytrue = test_sw2y_sarima.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw2y_sarima.yhat = test_sw2y_sarima.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 1Y
test_sw1y_sarima = df_values_sarima2
test_sw1y_sarima.index = dates
test_sw1y_sarima.ytrue = test_sw1y_sarima.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw1y_sarima.yhat = test_sw1y_sarima.yhat.apply(lambda x: float(x.strip("[]")))

# ---------------------//----------------------------------//---------------------------------------

# Divisão dos dados de teste para cada backtest de LSTM
# Expanding Window
test_ew_lstm = df_values_lstm1
test_ew_lstm.index = dates
test_ew_lstm.ytrue = test_ew_lstm.ytrue.apply(lambda x: float(x.strip("[]")))
test_ew_lstm.yhat = test_ew_lstm.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 2Y
test_sw2y_lstm = df_values_lstm2
test_sw2y_lstm.index = dates
test_sw2y_lstm.ytrue = test_sw2y_lstm.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw2y_lstm.yhat = test_sw2y_lstm.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 1Y
test_sw1y_lstm = df_values_lstm3
test_sw1y_lstm.index = dates
test_sw1y_lstm.ytrue = test_sw1y_lstm.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw1y_lstm.yhat = test_sw1y_lstm.yhat.apply(lambda x: float(x.strip("[]")))

# ---------------------//----------------------------------//---------------------------------------

# Divisão dos dados de teste para cada backtest de NeuralProphet
# Expanding Window
test_ew_np = df_values_np1
test_ew_np.index = dates
test_ew_np.ytrue = test_ew_np.ytrue.apply(lambda x: float(x.strip("[]")))
test_ew_np.yhat = test_ew_np.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 2Y
test_sw2y_np = df_values_np2
test_sw2y_np.index = dates
test_sw2y_np.ytrue = test_sw2y_np.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw2y_np.yhat = test_sw2y_np.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 1Y
test_sw1y_np = df_values_np3
test_sw1y_np.index = dates
test_sw1y_np.ytrue = test_sw1y_np.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw1y_np.yhat = test_sw1y_np.yhat.apply(lambda x: float(x.strip("[]")))

# ---------------------//----------------------------------//---------------------------------------

# Divisão dos dados de teste para cada backtest de CHRONOS
# Expanding Window
test_ew_cr = df_values_cr[:365]
test_ew_cr.index = dates
test_ew_cr.ytrue = test_ew_cr.ytrue.apply(lambda x: float(x.strip("[]")))
test_ew_cr.yhat = test_ew_cr.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 2Y
test_sw2y_cr = df_values_cr[365:730]
test_sw2y_cr.index = dates
test_sw2y_cr.ytrue = test_sw2y_cr.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw2y_cr.yhat = test_sw2y_cr.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 1Y
test_sw1y_cr = df_values_cr[730:]
test_sw1y_cr.index = dates
test_sw1y_cr.ytrue = test_sw1y_cr.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw1y_cr.yhat = test_sw1y_cr.yhat.apply(lambda x: float(x.strip("[]")))

abs_erros_arima_sw2y = analise_abs(test_sw2y_arima["ytrue"], test_sw2y_arima["yhat"])
abs_erros_arima_sw2y.biggest_errors(
    perc=5,
    output_dir=output_dir1,
    model_name="arima_sw2y",
    start="2019-01-01",
    end="2019-12-31",
)
abs_erros_arima_sw2y.plot_abs_errors(output_dir1, "ARIMA")

abs_erros_sarima_ew = analise_abs(test_ew_sarima["ytrue"], test_ew_sarima["yhat"])
abs_erros_sarima_ew.biggest_errors(
    perc=5,
    output_dir=output_dir1,
    model_name="sarima_ew",
    start="2019-01-01",
    end="2019-12-31",
)
abs_erros_sarima_ew.plot_abs_errors(output_dir1, "SARIMA")

abs_erros_lstm_sw2y = analise_abs(test_sw2y_lstm["ytrue"], test_sw2y_lstm["yhat"])
abs_erros_lstm_sw2y.biggest_errors(
    perc=5,
    output_dir=output_dir1,
    model_name="lstm_sw2y",
    start="2019-01-01",
    end="2019-12-31",
)
abs_erros_lstm_sw2y.plot_abs_errors(output_dir1, "LSTM")

abs_erros_np_ew = analise_abs(test_ew_np["ytrue"], test_ew_np["yhat"])
abs_erros_np_ew.biggest_errors(
    perc=5,
    output_dir=output_dir1,
    model_name="np_ew",
    start="2019-01-01",
    end="2019-12-31",
)
abs_erros_np_ew.plot_abs_errors(output_dir1, "NP")

abs_erros_cr_sw2y = analise_abs(test_sw2y_cr["ytrue"], test_sw2y_cr["yhat"])
abs_erros_cr_sw2y.biggest_errors(
    perc=5,
    output_dir=output_dir1,
    model_name="cr_sw2y",
    start="2019-01-01",
    end="2019-12-31",
)
abs_erros_cr_sw2y.plot_abs_errors(output_dir1, "CR")

models = {
    "ARIMA": analise_abs(test_sw2y_arima["ytrue"], test_sw2y_arima["yhat"]),
    "SARIMA": analise_abs(test_ew_sarima["ytrue"], test_ew_sarima["yhat"]),
    "LSTM": analise_abs(test_sw2y_lstm["ytrue"], test_sw2y_lstm["yhat"]),
    "NP": analise_abs(test_ew_np["ytrue"], test_ew_np["yhat"]),
    "CR": analise_abs(test_sw2y_cr["ytrue"], test_sw2y_cr["yhat"]),
}

# Parâmetros fixos
perc = 5  # percentual dos maiores erros
output_dir = output_dir1
start = "2019-01-01"
end = "2019-12-31"

# subplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(13, 10), sharex="col")


for i, (model_name, model_obj) in enumerate(models.items()):
    # Calcula os maiores erros e inicializa as variáveis internas
    model_obj.biggest_errors(
        perc=perc, output_dir=output_dir, model_name=model_name, start=start, end=end
    )

    # Coluna 1 - Série com APEs destacados
    ax1 = axes[i, 0]
    ax1.plot(model_obj.ytrue, label="ST original", linewidth=0.9)
    ax1.plot(
        model_obj.abs_errors,
        label="APEs",
        color="orange",
    )

    top_errors_dates = model_obj.abs_errors_desc.head(model_obj.n_errors).index
    top_errors_values = model_obj.abs_errors_desc.head(model_obj.n_errors).values

    ax1.scatter(
        top_errors_dates,
        top_errors_values,
        color="red",
        label="Top Erros",
        zorder=3,
        s=10,
    )
    for date in top_errors_dates:
        ax1.axvspan(date, date, color="red", alpha=0.4, lw=1.5)

    ax1.set_title(f"{model_name} - Série Original x APE", fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    if i == 0:
        ax1.legend(ncol=3, fontsize=12, loc="upper center")
    if i == len(models) - 1:
        ax1.set_xlabel("Ano", fontsize=12)

    # Coluna 2 - APEs ordenados decrescentemente
    ax2 = axes[i, 1]
    ax2.plot(model_obj.abs_errors_desc.values, linewidth=1)
    ax2.axvspan(0, model_obj.n_errors, color="red", alpha=0.3)
    ax2.set_title(f"{model_name} - Top {perc}% APEs Ordenados", fontsize=14)
    ax2.set_ylabel("APE (Em porcentagem)", fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 40, 65, 90, 115])
    ax2.tick_params(axis="y", labelsize=12)
    if i == len(models) - 1:
        ax2.set_xlabel("Índice ordenado", fontsize=12)
    ax2.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"Models_Top_Errors.pdf"), format="pdf")
plt.clf()
# plt.show()
