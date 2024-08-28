import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import os

import matplotlib.ticker

import dateutil.parser
import calplot
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# TODO: sex matplotlib text to black!
# TODO: latex and seaborn (googlit)

# Set matplotlib runtime configuration
# --------------------------------------------------------------------------------------
# http://aeturrell.com/2018/01/31/publication-quality-plots-in-python/
# https://scipy-cookbook.readthedocs.io/items/Matplotlib_LaTeX_Examples.html#Setting-Font-Sizes
# https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-style-sheets
# print(rcParams.keys())0

# NOTE: different palette
#  https://stackoverflow.com/questions/46148193/how-to-set-default-matplotlib-axis-colour-cycle
#  Seaborn in fact has six variations of matplotlib’s palette, called deep, muted,
#  pastel, bright, dark, and colorblind. These span a range of average luminance and
#  saturation values: https://seaborn.pydata.org/tutorial/color_palettes.html

# Computer Modern Sanf Serif
# https://seaborn.pydata.org/generated/seaborn.axes_style.html#seaborn.axes_style

# TODO: https://matplotlib.org/stable/users/explain/backends.html


# TODO:
#  1. Set sns grid manually (color)
#  2. Set palette (see note above)
#  3. Maybe let even the black from seaborn in text..
# sns.set_theme(style="whitegrid", palette="pastel")
class plot:
    sns.set_theme()
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind")

    SIZE = 8
    COLOR = "black"
    params = {
        "backend": "ps",
        # "backend": "Agg",
        "axes.titlesize": SIZE,
        "axes.labelsize": SIZE,
        "font.size": SIZE,
        # "text.fontsize": SIZE,
        "legend.fontsize": SIZE,
        "xtick.labelsize": SIZE,
        "ytick.labelsize": SIZE,
        "text.usetex": True,
        "font.family": "serif",
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
    }

    rcParams.update(params)

    plt.rc("font", size=SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SIZE)  # legend fontsize
    plt.rc("figure", titlesize=SIZE)  # fontsize of the figure title

    # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    from matplotlib import font_manager

    ticksfont = font_manager.FontProperties(
        family="sans-serif",
        style="normal",
        size=10,
        # weight="normal", stretch='normal'
    )

    # LaTex
    # --------------------------------------------------------------------------------------
    # The column width is: 455.24411pt
    # The text width is: 455.24411pt
    # The text height is: 702.78308pt
    #
    # The paper width is: 597.50787pt
    # The paper height is: 845.04684pt

    # LaTex
    # \message{The column width is: \the\columnwidth}
    # \message{The paper width is: \the\paperwidth}
    # \message{The paper height is: \the\paperheight}
    # \message{The text height is: \the\textheight}
    # \message{The text width is: \the\textwidth}

    textwidth = 455.24411  # Value given by Latex
    textheigth = 702.78308  # Value given by Latex

    class OOMFormatter(matplotlib.ticker.ScalarFormatter):
        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            matplotlib.ticker.ScalarFormatter.__init__(
                self, useOffset=offset, useMathText=mathText
            )

        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = self.oom

        def _set_format(self, vmin=None, vmax=None):
            self.format = self.fformat
            if self._useMathText:
                self.format = r"$\mathdefault{%s}$" % self.format

    def setup(xstart=None, xend=None, nrows=2, xfmt=mdates.DateFormatter("%y.%m.%d")):
        """Organize plot style and create matplotlib figure and axis.

        :param xstart: left x axis limit
        :param xend: right x axis limit
        :param nrows: number of rows of the subplot grid
        :param xfmt: formatter of the x axis major ticker
        :return: a tuple with matplotlib figure and axis
        """
        # plt.style.use("ggplot")
        # sns.set_style("whitegrid")

        fig, axs = plt.subplots(nrows)

        # Check if it's an iterable since nrows can be one and axs will be a single object
        for axis in axs if hasattr(axs, "__iter__") else [axs]:
            if xfmt:
                axis.xaxis.set_major_formatter(xfmt)
            if xstart and xend:
                axis.set_xlim(xstart, xend)

        return fig, axs

    def save(pdf, filename):
        if not pdf:
            # TODO: save directly and use tight
            # plt.savefig(filename)
            with PdfPages(filename) as pdf:
                wrapup(pdf)
        else:
            wrapup(pdf)

    def wrapup(pdf=None, show=False):
        """Finalize current figure. It will clear and close after show and/or save it.

        bbox: https://stackoverflow.com/a/11847260/14113878

        :param pdf: matplotlib PdfPages object to save current figure
        :param show: display current figure
        :param bbox:
        """
        if pdf:
            pdf.savefig(bbox_inches="tight")
        if show:
            plt.show()

        plt.clf()
        plt.close("all")

    def get_figsize(columnwidth, wf=0.5, hf=(5.0**0.5 - 1.0) / 2.0):
        # """Parameters:
        # - wf [float]: width fraction in columnwidth units
        # - hf [float]: height fraction in columnwidth units. Set by default to golden ratio.
        # - columnwidth [float]: width of the column in latex. Get this from LaTeX using the
        # follwoing command: \showthe\columnwidth

        # Returns: [fig_width, fig_height]: that should be given to matplotlib
        # """
        fig_width_pt = columnwidth * wf
        inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        fig_width = fig_width_pt * inches_per_pt  # Width in inches
        fig_height = fig_width * hf  # Height in inches

        return [fig_width, fig_height]

    def zoomin(y, xrange, yrange, axs):
        """
        >>> x1 = df.index[len(df) // 2 - 65]
        >>> x2 = df.index[len(df) // 2 + 65]
        >>> y1 = df[column][len(df) // 2] - 1.5
        >>> y2 = df[column][len(df) // 2] + 1.5
        >>> plot.zoomin(df[column], (x1, x2), (y1, y2), axs)
        """

        axins = zoomed_inset_axes(axs, zoom=2, loc="upper right")
        axins.plot(y)
        axins.set_xlim(*xrange)
        axins.set_ylim(*yrange)

        plt.xticks(visible=False)
        plt.yticks(visible=False)

        mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        plt.draw()

    # ======================================================================================

    def yearly(df, columns, years, pdf):
        # Plot each year
        for year in years:
            prox = str(int(year) + 1)
            tmp = df[df.index > dateutil.parser.parse(f"{year}-01-01")]
            tmp = tmp[tmp.index < dateutil.parser.parse(f"{prox}-01-01")]

            fig, axs = setup(nrows=1)
            tmp.plot(
                kind="line",
                style=".-",
                title=f"{year}",
                y=columns,
                use_index=True,
                ax=axs,
            )

            wrapup(pdf, False)

        # Plot all years
        fig, axs = setup(nrows=1)
        df.plot(
            kind="line",
            style=".-",
            title=f"{years[0]} to {years[-1]}",
            y=columns,
            use_index=True,
            ax=axs,
        )
        wrapup(pdf, False)

    def correlation(df, method, pdf, xticks=False):
        corr = df.corr(method).sort_values("water_produced", ascending=True)
        corr = corr.reindex(corr.index, axis=1)

        corrmatrix(corr, f"{method.capitalize()} Correlation\n", pdf, xticks)

        # df = df.dropna()
        #
        # g = sns.pairplot(df, hue="dayofweek", diag_kind="kde")
        # for ax in g.axes.flatten():
        #     ax.set_xlabel(ax.get_xlabel(), rotation=45)
        #     ax.set_ylabel(ax.get_ylabel(), rotation=45)
        #     ax.yaxis.get_label().set_horizontalalignment("right")
        #
        # plt.yticks(rotation=45)
        # plot.wrapup(pdf, False, "tight")
        #
        # g = sns.pairplot(df, diag_kind="kde")
        # for ax in g.axes.flatten():
        #     ax.set_xlabel(ax.get_xlabel(), rotation=90)
        #     ax.set_ylabel(ax.get_ylabel(), rotation=0)
        #     ax.yaxis.get_label().set_horizontalalignment("right")
        #
        # plt.yticks(rotation=45)
        # plt.title("Pair Plot")
        # plot.wrapup(pdf, False, "tight")

    def pbc(df, continuous, pdf, xticks=False):
        # Assume all other columns besides from `continuous` are dichotomous (aka binary)
        # len(df.columns)
        corr = pd.DataFrame(index=df.columns, columns=df.columns)

        y = df[continuous]
        for name, values in df.iteritems():
            if name == continuous:
                continue

            x = values
            # x: binary variable (boolean)
            # y: continuous variable
            coef, p = scipy.stats.pointbiserialr(x, y)

            corr.at[name, continuous] = coef

        for col in corr.columns:
            corr[col] = pd.to_numeric(corr[col], errors="coerce")

        corr = corr.sort_values(by="water_produced", ascending=True)
        corr = corr.reindex(corr.index, axis=1)

        corrmatrix(corr, f"PBS Correlation\n", pdf, xticks)

    def corrmatrix(corr, title, pdf, xticks=False):
        sns.heatmap(
            corr,
            square=True,
            fmt=".2f",
            cmap=sns.color_palette("vlag", as_cmap=True),
            annot=True,
            mask=corr.isnull(),  # annot_kws={"size": 8}
        )

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if not xticks:
            plt.xticks([], [])
        plt.title(title)

        # wrapup(pdf, False, "tight")

    def calhm(dates, title, pdf):
        calplot.calplot(
            dates,
            cmap="inferno",
            colorbar=False,
            linewidth=3,
            edgecolor="gray",
            figsize=rcParams["figure.figsize"],
            suptitle=f"\n{title}",
        )

        wrapup(pdf, False)


"""#Divisão dos dados de reais e teste para cada modelo"""

# df_metrics_lr = pd.read_csv('/content/drive/MyDrive/Dataset/Resultados/output/metrics-LR_2024-04-23-17-15-26.csv')
# df_metrics_arima = pd.read_csv('/content/drive/MyDrive/PPGESE/ARTIGO/Results/metrics-arima_2024-05-20-15-38-45.csv')
# df_metrics_sarima = pd.read_csv('/content/drive/MyDrive/PPGESE/ARTIGO/Results/metrics-sarima_2024-05-22-09-36-40.csv')
# df_metrics_sarima2 = pd.read_csv('/content/drive/MyDrive/PPGESE/ARTIGO/Results/metrics-sarima_2024-05-25-01-55-14.csv')
# df_metrics_lstm = pd.read_csv('/content/drive/MyDrive/Dataset/Resultados/output/metrics-LSTM_2024-04-23-17-22-19.csv')

# df_values_lr = pd.read_csv('/content/drive/MyDrive/Dataset/Resultados/output_ytrue_yhat/results_LR-2024-04-23-17-15-26.csv')
df_values_arima = pd.read_csv("output_ytrue_yhat/results_arima-2024-05-20-15-38-45.csv")
df_values_sarima = pd.read_csv(
    "output_ytrue_yhat/results_sarima-2024-05-22-09-36-40.csv"
)
df_values_sarima2 = pd.read_csv(
    "output_ytrue_yhat/results_sarima-2024-05-25-01-55-14.csv"
)
df_values_lstm = pd.read_csv("output_ytrue_yhat/results_lstm-2024-06-27-10-33-50.csv")
df_values_lstm2 = pd.read_csv("output_ytrue_yhat/results_lstm-2024-06-28-23-29-24.csv")
df_values_lstm3 = pd.read_csv("output_ytrue_yhat/results_lstm-2024-06-30-23-01-04.csv")

dates = pd.date_range(start="2019-01-01", end="2019-12-31")

# Divisão dos dados de teste para cada backtest de LR
# Expanding Window
# test_ew_lr = df_values_lr[:365]
# test_ew_lr.index = dates
# test_ew_lr.ytrue = test_ew_lr.ytrue.apply(lambda x: float(x.strip('[]')))
# test_ew_lr.yhat = test_ew_lr.yhat.apply(lambda x: float(x.strip('[]')))

# # Sliding Window 2Y
# test_sw2y_lr = df_values_lr[365:730]
# test_sw2y_lr.index = dates
# test_sw2y_lr.ytrue = test_sw2y_lr.ytrue.apply(lambda x: float(x.strip('[]')))
# test_sw2y_lr.yhat = test_sw2y_lr.yhat.apply(lambda x: float(x.strip('[]')))

# # Sliding Window 1Y
# test_sw1y_lr = df_values_lr[730:]
# test_sw1y_lr.index = dates
# test_sw1y_lr.ytrue = test_sw1y_lr.ytrue.apply(lambda x: float(x.strip('[]')))
# test_sw1y_lr.yhat = test_sw1y_lr.yhat.apply(lambda x: float(x.strip('[]')))

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
test_ew_sarima = df_values_sarima[:365]
test_ew_sarima.index = dates
test_ew_sarima.ytrue = test_ew_sarima.ytrue.apply(lambda x: float(x.strip("[]")))
test_ew_sarima.yhat = test_ew_sarima.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 2Y
test_sw2y_sarima = df_values_sarima[365:730]
test_sw2y_sarima.index = dates
test_sw2y_sarima.ytrue = test_sw2y_sarima.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw2y_sarima.yhat = test_sw2y_sarima.yhat.apply(lambda x: float(x.strip("[]")))

# Sliding Window 1Y
# test_sw1y_sarima = df_values_sarima[730:]
test_sw1y_sarima = df_values_sarima2[:365]
test_sw1y_sarima.index = dates
test_sw1y_sarima.ytrue = test_sw1y_sarima.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw1y_sarima.yhat = test_sw1y_sarima.yhat.apply(lambda x: float(x.strip("[]")))

# ---------------------//----------------------------------//---------------------------------------

# Divisão dos dados de teste para cada backtest de LSTM
# Expanding Window
test_ew_lstm = df_values_lstm[:365]
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

"""#Separação dos dados por mês"""

# Separar os dados de ytrue e yhat por mês
# Atualmente não utilizado
# ------------------------------------------------------------------------------
# test_ew_lr['month'] = test_ew_lr.index.month
# pivot_test_ew_lr = test_ew_lr.pivot_table(index=test_ew_lr.index.day, columns=test_ew_lr.index.month, values=['ytrue', 'yhat'])

test_sw2y_arima["month"] = test_sw2y_arima.index.month
pivot_test_sw2y_arima = test_sw2y_arima.pivot_table(
    index=test_sw2y_arima.index.day,
    columns=test_sw2y_arima.index.month,
    values=["ytrue", "yhat"],
)

test_ew_sarima["month"] = test_ew_sarima.index.month
pivot_test_ew_sarima = test_ew_sarima.pivot_table(
    index=test_ew_sarima.index.day,
    columns=test_ew_sarima.index.month,
    values=["ytrue", "yhat"],
)

# test_ew_lstm['month'] = test_ew_lstm.index.month
# pivot_test_ew_lstm = test_ew_lstm.pivot_table(index=test_ew_lstm.index.day, columns=test_ew_lstm.index.month, values=['ytrue', 'yhat'])

from matplotlib.ticker import MultipleLocator

# Configuração para usar Times New Roman e aumentar o tamanho da fonte
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

# Lista de dataframes e nomes dos modelos
dfs = [test_sw2y_arima, test_ew_sarima, test_sw2y_lstm]
model_names = ["ARIMA-SW2Y", "SARIMA-EW", "LSTM-SW2Y"]

# Tamanho da figura
figsize = (8.27, 11.69)
fig, axes = plt.subplots(3, 1, figsize=figsize)

# Plotando cada modelo
for i, (dfpred, model_name) in enumerate(zip(dfs, model_names)):
    ax = axes[i]
    sns.regplot(
        x="ytrue",
        y="yhat",
        data=dfpred,
        ax=ax,
        scatter=True,
        label=model_name,
        scatter_kws={"alpha": 0.5},
    )

    # corr = dfpred[['ytrue', 'yhat']].corr().iloc[1, 0]
    # corr = round(corr, 2)
    r2 = round(r2_score(dfpred["ytrue"], dfpred["yhat"]), 3)
    ax.set_title(f"{model_name}: $R^2$={r2}")

    # print(r2)
    # sw2y = 0.7528520291894026
    # ew = 0.7451190210492027


# Ajustar os subplots
for i, ax in enumerate(axes):
    ax.yaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.xaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    ax.set(xlabel="Real")
    ax.set(ylabel="Previsto")

plt.tight_layout()

output_dir = "comparativos"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "realxpred_r2.pdf")
plt.savefig(output_file)

# print(f"Gráfico salvo como {filename}")
