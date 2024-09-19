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

    def biggest_errors(self, perc, start, end):
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

        print("Number of errors: ", len(self.abs_errors_desc[: self.n_errors]))
        print(
            f"The top {self.perc}% biggest errors are: ",
            self.abs_errors_desc[: self.n_errors],
        )

    def plot_abs_errors(self, output_dir, model_name):
        plt.figure(figsize=(11.69, 8.27))
        plt.plot(self.abs_errors_desc.values)
        plt.axvspan(0, self.n_errors, color="red", alpha=0.3)
        plt.title(f"The top {self.perc}% errors", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_top_errors.pdf"), format="pdf"
        )
        plt.clf()

        plt.figure(figsize=(11.69, 8.27))
        plt.plot(self.ytrue, label="Original time series")
        plt.plot(self.abs_errors, label="Absolute percentage errors")

        top_errors_dates = self.abs_errors_desc.head(self.n_errors).index
        top_errors_values = self.abs_errors_desc.head(self.n_errors).values

        for date in top_errors_dates:
            plt.axvspan(date, date, color="red", alpha=0.4, lw=1.5)

        plt.scatter(
            top_errors_dates,
            top_errors_values,
            color="red",
            label="Top Errors",
            zorder=3,
        )

        plt.legend(fontsize=12)
        plt.title("Original ST x Absolute percentage Errors", fontsize=16)
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

df_arima = pd.read_csv("output_ytrue_yhat/results_arima-2024-05-20-15-38-45.csv")
df_arima = pd.DataFrame(df_arima)
df_arima["ytrue"] = (
    df_arima["ytrue"].str.replace(r"[\[\]]", "", regex=True).astype(float)
)
df_arima["yhat"] = df_arima["yhat"].str.replace(r"[\[\]]", "", regex=True).astype(float)

arima_ytrue_sw2y = df_arima["ytrue"][365:730]
arima_yhat_sw2y = df_arima["yhat"][365:730]

abs_erros_arima_sw2y = analise_abs(arima_ytrue_sw2y, arima_yhat_sw2y)
abs_erros_arima_sw2y.biggest_errors(perc=5, start="2019-01-01", end="2019-12-31")
abs_erros_arima_sw2y.plot_abs_errors(output_dir1, "Arima")
