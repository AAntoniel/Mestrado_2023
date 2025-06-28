import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
import matplotlib
import os

matplotlib.use("TkAgg")
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

import matplotlib.ticker


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


# Diretório
output_dir = "comparativos"
os.makedirs(output_dir, exist_ok=True)

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

# -------------------------------- COMPARAÇÃO DOS MELHORES MODELOS MÊS A MÊS --------------------------------

## Calculo dos ABSs por mês
arima_abs = []
sarima_abs = []
lstm_abs = []
np_abs = []
cr_abs = []

abs_arima = np.abs(test_sw2y_arima["ytrue"] - test_sw2y_arima["yhat"])
arima_abs.append(abs_arima)

abs_sarima = np.abs(test_ew_sarima["ytrue"] - test_ew_sarima["yhat"])
sarima_abs.append(abs_sarima)

abs_lstm = np.abs(test_sw2y_lstm["ytrue"] - test_sw2y_lstm["yhat"])
lstm_abs.append(abs_lstm)

abs_np = np.abs(test_ew_np["ytrue"] - test_ew_np["yhat"])
np_abs.append(abs_np)

abs_cr = np.abs(test_sw2y_cr["ytrue"] - test_sw2y_cr["yhat"])
cr_abs.append(abs_cr)

df = pd.concat(arima_abs + sarima_abs + lstm_abs + np_abs + cr_abs)
df = pd.DataFrame(df)
df.columns = ["abs"]

meses_traduzidos = {
    "January": "Jan",
    "February": "Fev",
    "March": "Mar",
    "April": "Abr",
    "May": "Mai",
    "June": "Jun",
    "July": "Jul",
    "August": "Ago",
    "September": "Set",
    "October": "Out",
    "November": "Nov",
    "December": "Dez",
}
df["Mes"] = df.index.strftime("%B").map(meses_traduzidos)

Models = ["ARIMA-SW2Y", "SARIMA-EW", "LSTM-SW2Y", "NP-EW", "CR-SW2Y"]
n = len(df) // len(Models)
modelos_repetidos = [model for model in Models for _ in range(n)]
modelos_repetidos += Models[: (len(df) - len(modelos_repetidos))]
df["Modelos"] = modelos_repetidos

palette = sns.color_palette("colorblind", n_colors=5)
meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "#4c4c4c",
    "markersize": "5",
}

figsize = get_figsize(textwidth, wf=1.0)

plt.figure(figsize=(11.69, 8.27))
sns.boxplot(
    x="Mes",
    y="abs",
    hue="Modelos",
    data=df,
    palette=palette,
    showmeans=True,
    meanprops=meanprops,
    linewidth=2.0,
)
plt.title("Comparação dos Modelos por Mês", fontsize=18)
plt.xlabel("Mês", fontsize=16)
plt.ylabel("ABS", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title="Modelos", title_fontsize=16, fontsize=14)
plt.tight_layout()
# plt.show()
output_file = os.path.join(output_dir, "boxplot_modelos_por_mes.pdf")
plt.savefig(output_file)

# -------------------------------- COMPARAÇÃO ENTRE OS VALORES REAIS E PREVISTOS PARA TODO O PERÍODO --------------------------------

real = pd.read_csv("data/dataset.csv")
real = real.dropna()
real["timestamp"] = pd.to_datetime(real["timestamp"])
real.set_index("timestamp", inplace=True)

plt.style.use("seaborn-v0_8-colorblind")
plt.figure(figsize=(12, 6))
# plt.plot(real.water_produced[1095:], label="REAL", color="#619CFF")
plt.plot(real.water_produced[1095:], label="REAL", color="#000000")
plt.plot(test_sw2y_arima.yhat, label="ARIMA-SW2Y", color="#E69F00")
plt.plot(test_ew_sarima.yhat, label="SARIMA-EW", color="#00BA38")
plt.plot(test_sw2y_lstm.yhat, label="LSTM-SW2Y", color="#D55E00")
plt.plot(test_ew_np.yhat, label="NP-EW", color="#CC79A7")
# plt.plot(test_sw2y_cr.yhat, label="CR-SW2Y", color="#F0E442")
plt.plot(test_sw2y_cr.yhat, label="CR-SW2Y", color="#619CFF")
plt.legend(fontsize=12, ncol=6, loc="upper center")
plt.xlabel("2019", fontsize=16)
plt.ylabel("Demanda de Água (m³)", fontsize=16)
plt.title("Comparação entre valores reais e previstos para o ano de 2019", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
# plt.show()
output_file2 = os.path.join(output_dir, "comparacao_reais_previstos_total.pdf")
plt.savefig(output_file2)

# -------------------------------- COMPARAÇÃO ENTRE OS VALORES REAIS E PREVISTOS PARA MARÇO, SETEMBRO E DEZEMBRO --------------------------------

# Períodos

periods = {
    "Março": ("2019-03-01", "2019-03-31"),
    "Setembro": ("2019-09-01", "2019-09-30"),
    "Dezembro": ("2019-12-01", "2019-12-31"),
}

# Modelos

models = {
    "Real": ("#619CFF", real.water_produced),
    "ARIMA": ("#E69F00", test_sw2y_arima.yhat),
    "SARIMA": ("#00BA38", test_ew_sarima.yhat),
    "LSTM": ("#D55E00", test_sw2y_lstm.yhat),
    "NP": ("#CC79A7", test_ew_np.yhat),
    "CR": ("#F0E442", test_sw2y_cr.yhat),
}

fig, axes = plt.subplots(len(periods), 1, figsize=(11.69, 8.27))

for i, (month, (start, end)) in enumerate(periods.items()):
    for model, (color, data) in models.items():
        subplot = data.loc[start:end]
        if not subplot.empty:
            subplot.plot(
                ax=axes[i], color=color, fontsize=10, label=model if i == 0 else None
            )
        axes[i].set_title(month, fontsize=10)
    axes[0].legend(
        models.keys(),
        fontsize=10,
        loc="upper right",
        # bbox_to_anchor=(0.5, -0.1),
        ncol=len(models),
    )

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

output_file3 = os.path.join(output_dir, "comparacao_realpred_mar_sep_dec.pdf")
plt.savefig(output_file3)

# ----------- COMPARAÇÃO ENTRE OS VALORES REAIS E PREVISTOS PARA TODO O PERÍODO SEGMENTADOS --------------------------------

# Modelos

models = {
    "Real": ("#619CFF", real.water_produced[1095:]),
    "ARIMA": ("#E69F00", test_sw2y_arima.yhat),
    "SARIMA": ("#00BA38", test_ew_sarima.yhat),
    "LSTM": ("#D55E00", test_sw2y_lstm.yhat),
    "NP": ("#CC79A7", test_ew_np.yhat),
    "CR": ("#F0E442", test_sw2y_cr.yhat),
}

fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27), sharex=True, sharey=True)
axes = axes.flatten()
for i in range(len(models)):
    real.water_produced[1095:].plot(ax=axes[i], color="#619CFF")
    if i == 0:
        axes[i].set_title("REAL", fontsize=14)
        axes[i].tick_params(axis="y", labelsize=14)

for i, (model, (color, data)) in enumerate(models.items()):
    if i == 0:
        continue

    subplot = data
    if not subplot.empty:
        subplot.plot(
            ax=axes[i], color=color, fontsize=12, label=model if i == 0 else None
        )
    axes[i].set_title(model, fontsize=14)
# # axes[0].legend(
#     models.keys(),
#     fontsize=10,
#     loc="upper right",
#     # bbox_to_anchor=(0.5, -0.1),
#     # ncol=len(models),
# )

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

output_file3 = os.path.join(output_dir, "comparacao_realpred_segmentado.pdf")
plt.savefig(output_file3)

# color="#619CFF" real
# color="#E69F00" arima
# color="#00BA38" sarima
# color="#D55E00" lstm
# color="#CC79A7" np
# color="#F0E442" cr

# -------------------------------- COMPARAÇÃO DOS MODELOS MÊS A MÊS EW --------------------------------

## Calculo dos ABSs por mês
arima_ew_abs = []
sarima_ew_abs = []
lstm_ew_abs = []
np_ew_abs = []
cr_ew_abs = []

abs_ew_arima = np.abs(test_ew_arima["ytrue"] - test_ew_arima["yhat"])
arima_ew_abs.append(abs_ew_arima)

abs_ew_sarima = np.abs(test_ew_sarima["ytrue"] - test_ew_sarima["yhat"])
sarima_ew_abs.append(abs_ew_sarima)

abs_ew_lstm = np.abs(test_ew_lstm["ytrue"] - test_ew_lstm["yhat"])
lstm_ew_abs.append(abs_ew_lstm)

abs_ew_np = np.abs(test_ew_np["ytrue"] - test_ew_np["yhat"])
np_ew_abs.append(abs_ew_np)

abs_ew_cr = np.abs(test_ew_cr["ytrue"] - test_ew_cr["yhat"])
cr_ew_abs.append(abs_ew_cr)

df_ew = pd.concat(arima_ew_abs + sarima_ew_abs + lstm_ew_abs + np_ew_abs + cr_ew_abs)
df_ew = pd.DataFrame(df_ew)
df_ew.columns = ["abs"]
df_ew["Mes"] = df_ew.index.strftime("%B").str.capitalize()

Models = ["ARIMA-EW", "SARIMA-EW", "LSTM-EW", "NP-EW", "CR-EW"]
n = len(df_ew) // len(Models)
modelos_repetidos = [model for model in Models for _ in range(n)]
modelos_repetidos += Models[: (len(df_ew) - len(modelos_repetidos))]
df_ew["Modelos"] = modelos_repetidos

palette = sns.color_palette("colorblind", n_colors=5)
meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "#4c4c4c",
    "markersize": "3",
}

figsize = get_figsize(textwidth, wf=1.0)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Mes",
    y="abs",
    hue="Modelos",
    data=df_ew,
    palette=palette,
    showmeans=True,
    meanprops=meanprops,
)
plt.title("Comparação dos Modelos EW por Mês")
plt.xlabel("Mês")
plt.ylabel("ABS")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
output_file4 = os.path.join(output_dir, "boxplot_modelos_EW_por_mes.pdf")
plt.savefig(output_file4)

# -------------------------------- COMPARAÇÃO DOS MODELOS MÊS A MÊS SW1Y --------------------------------

## Calculo dos ABSs por mês
arima_sw1y_abs = []
sarima_sw1y_abs = []
lstm_sw1y_abs = []
np_sw1y_abs = []
cr_sw1y_abs = []

abs_sw1y_arima = np.abs(test_sw1y_arima["ytrue"] - test_sw1y_arima["yhat"])
arima_sw1y_abs.append(abs_sw1y_arima)

abs_sw1y_sarima = np.abs(test_sw1y_sarima["ytrue"] - test_sw1y_sarima["yhat"])
sarima_sw1y_abs.append(abs_sw1y_sarima)

abs_sw1y_lstm = np.abs(test_sw1y_lstm["ytrue"] - test_sw1y_lstm["yhat"])
lstm_sw1y_abs.append(abs_sw1y_lstm)

abs_sw1y_np = np.abs(test_sw1y_np["ytrue"] - test_sw1y_np["yhat"])
np_sw1y_abs.append(abs_sw1y_np)

abs_sw1y_cr = np.abs(test_sw1y_cr["ytrue"] - test_sw1y_cr["yhat"])
cr_sw1y_abs.append(abs_sw1y_cr)

df_sw1y = pd.concat(
    arima_sw1y_abs + sarima_sw1y_abs + lstm_sw1y_abs + np_sw1y_abs + cr_sw1y_abs
)
df_sw1y = pd.DataFrame(df_sw1y)
df_sw1y.columns = ["abs"]
df_sw1y["Mes"] = df_sw1y.index.strftime("%B").str.capitalize()

Models = ["ARIMA-SW1Y", "SARIMA-SW1Y", "LSTM-SW1Y", "NP-SW1Y", "CR-SW1Y"]
n = len(df_sw1y) // len(Models)
modelos_repetidos = [model for model in Models for _ in range(n)]
modelos_repetidos += Models[: (len(df_sw1y) - len(modelos_repetidos))]
df_sw1y["Modelos"] = modelos_repetidos

# Plotando o boxplot
palette = sns.color_palette("colorblind", n_colors=5)
meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "#4c4c4c",
    "markersize": "3",
}

figsize = get_figsize(textwidth, wf=1.0)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Mes",
    y="abs",
    hue="Modelos",
    data=df_sw1y,
    palette=palette,
    showmeans=True,
    meanprops=meanprops,
)
plt.title("Comparação dos Modelos SW1Y por Mês")
plt.xlabel("Mês")
plt.ylabel("ABS")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
output_file5 = os.path.join(output_dir, "boxplot_modelos_SW1Y_por_mes.pdf")
plt.savefig(output_file5)

# -------------------------------- COMPARAÇÃO DOS MODELOS MÊS A MÊS SW2Y --------------------------------

## Calculo dos ABSs por mês
arima_sw2y_abs = []
sarima_sw2y_abs = []
lstm_sw2y_abs = []
np_sw2y_abs = []
cr_sw2y_abs = []

abs_sw2y_arima = np.abs(test_sw2y_arima["ytrue"] - test_sw2y_arima["yhat"])
arima_sw2y_abs.append(abs_sw2y_arima)

abs_sw2y_sarima = np.abs(test_sw2y_sarima["ytrue"] - test_sw2y_sarima["yhat"])
sarima_sw2y_abs.append(abs_sw2y_sarima)

abs_sw2y_lstm = np.abs(test_sw2y_lstm["ytrue"] - test_sw2y_lstm["yhat"])
lstm_sw2y_abs.append(abs_sw2y_lstm)

abs_sw2y_np = np.abs(test_sw2y_np["ytrue"] - test_sw2y_np["yhat"])
np_sw2y_abs.append(abs_sw2y_np)

abs_sw2y_cr = np.abs(test_sw2y_cr["ytrue"] - test_sw2y_cr["yhat"])
cr_sw2y_abs.append(abs_sw2y_cr)

df_sw2y = pd.concat(
    arima_sw2y_abs + sarima_sw2y_abs + lstm_sw2y_abs + np_sw2y_abs + cr_sw2y_abs
)
df_sw2y = pd.DataFrame(df_sw2y)
df_sw2y.columns = ["abs"]
df_sw2y["Mes"] = df_sw2y.index.strftime("%B").str.capitalize()

Models = ["ARIMA-SW2Y", "SARIMA-SW2Y", "LSTM-SW2Y", "NP-SW2Y", "CR-SW2Y"]
n = len(df_sw2y) // len(Models)
modelos_repetidos = [model for model in Models for _ in range(n)]
modelos_repetidos += Models[: (len(df_sw2y) - len(modelos_repetidos))]
df_sw2y["Modelos"] = modelos_repetidos

# boxplot
palette = sns.color_palette("colorblind", n_colors=5)
meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "#4c4c4c",
    "markersize": "3",
}

figsize = get_figsize(textwidth, wf=1.0)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Mes",
    y="abs",
    hue="Modelos",
    data=df_sw2y,
    palette=palette,
    showmeans=True,
    meanprops=meanprops,
)
plt.title("Comparação dos Modelos SW2Y por Mês")
plt.xlabel("Mês")
plt.ylabel("ABS")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
output_file6 = os.path.join(output_dir, "boxplot_modelos_SW2Y_por_mes.pdf")
plt.savefig(output_file6)

# -------------------------------- COMPARAÇÃO DOS MODELOS SEMANA A SEMANA MARÇO --------------------------------

df_weeks1 = df.loc[df.index.month == 3]
df_weeks1["Semana"] = df_weeks1.index.to_series().dt.isocalendar().week
# print(df_weeks1)

df_weeks1["Semana"] = df_weeks1["Semana"].astype(str)
print(df_weeks1)

# boxplot
palette = sns.color_palette("colorblind", n_colors=5)
meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "#4c4c4c",
    "markersize": "3",
}

figsize = get_figsize(textwidth, wf=1.0)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Semana",
    y="abs",
    hue="Modelos",
    data=df_weeks1,
    palette=palette,
    showmeans=True,
    meanprops=meanprops,
)
plt.title("Comparação dos melhores modelos no mês de Março (por semanas)")
plt.xlabel("Mês")
plt.ylabel("ABS")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
output_file7 = os.path.join(output_dir, "boxplot_modelos_semanal_marco.pdf")
plt.savefig(output_file7)

# -------------------------------- COMPARAÇÃO DOS MODELOS SEMANA A SEMANA SETEMBRO --------------------------------

df_weeks2 = df.loc[df.index.month == 9]
df_weeks2["Semana"] = df_weeks2.index.to_series().dt.isocalendar().week
# print(df_weeks2)

df_weeks2 = df_weeks2.drop(index=["2019-09-01", "2019-09-30"])

df_weeks2["Semana"] = df_weeks2["Semana"].astype(str)
print(df_weeks2)

# boxplot
palette = sns.color_palette("colorblind", n_colors=5)
meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "#4c4c4c",
    "markersize": "3",
}

figsize = get_figsize(textwidth, wf=1.0)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Semana",
    y="abs",
    hue="Modelos",
    data=df_weeks2,
    palette=palette,
    showmeans=True,
    meanprops=meanprops,
)
plt.title("Comparação dos melhores modelos no mês de Setembro (por semanas)")
plt.xlabel("Mês")
plt.ylabel("ABS")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
output_file8 = os.path.join(output_dir, "boxplot_modelos_semanal_setembro.pdf")
plt.savefig(output_file8)

# -------------------------------- COMPARAÇÃO DOS MODELOS SEMANA A SEMANA DEZEMBRO --------------------------------

df_weeks3 = df.loc[df.index.month == 12]
df_weeks3["Semana"] = df_weeks3.index.to_series().dt.isocalendar().week

df_weeks3["Semana"] = df_weeks3["Semana"].astype(str)

df_weeks3 = df_weeks3.drop(index=["2019-12-01"])
print(df_weeks3)

# boxplot
palette = sns.color_palette("colorblind", n_colors=5)
meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "#4c4c4c",
    "markersize": "3",
}

figsize = get_figsize(textwidth, wf=1.0)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Semana",
    y="abs",
    hue="Modelos",
    data=df_weeks3,
    palette=palette,
    showmeans=True,
    meanprops=meanprops,
)
plt.title("Comparação dos melhores modelos no mês de dezembro (por semanas)")
plt.xlabel("Mês")
plt.ylabel("ABS")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
output_file9 = os.path.join(output_dir, "boxplot_modelos_semanal_dezembro.pdf")
plt.savefig(output_file9)

# -------------------------------- COMPARAÇÃO ENTRE OS VALORES REAIS E PREVISTOS SEGMENTADO POR MESES ----------------------------------
# Períodos

periods = {
    "Janeiro": ("2019-01-01", "2019-01-31"),
    "Fevereiro": ("2019-02-01", "2019-02-28"),
    "Março": ("2019-03-01", "2019-03-31"),
    "Abril": ("2019-04-01", "2019-04-30"),
    "Maio": ("2019-05-01", "2019-05-31"),
    "Junho": ("2019-06-01", "2019-06-30"),
    "Julho": ("2019-07-01", "2019-07-31"),
    "Agosto": ("2019-08-01", "2019-08-31"),
    "Setembro": ("2019-09-01", "2019-09-30"),
    "Outubro": ("2019-10-01", "2019-10-31"),
    "Novembro": ("2019-11-01", "2019-11-30"),
    "Dezembro": ("2019-12-01", "2019-12-31"),
}

# Modelos

# models = {
#     "Real": ("#619CFF", real.water_produced),
#     "ARIMA": ("#E69F00", test_sw2y_arima.yhat),
#     "SARIMA": ("#00BA38", test_ew_sarima.yhat),
#     "LSTM": ("#D55E00", test_sw2y_lstm.yhat),
#     "NP": ("#CC79A7", test_ew_np.yhat),
#     "CR": ("#F0E442", test_sw2y_cr.yhat),
# }

models = {
    "Real": ("#000000", real.water_produced),
    "ARIMA": ("#E69F00", test_sw2y_arima.yhat),
    "SARIMA": ("#00BA38", test_ew_sarima.yhat),
    "LSTM": ("#D55E00", test_sw2y_lstm.yhat),
    "NP": ("#CC79A7", test_ew_np.yhat),
    "CR": ("#619CFF", test_sw2y_cr.yhat),
}

fig, axes = plt.subplots(4, 3, figsize=(11.69, 8.27), sharex=True, sharey=True)

dias_do_mes = list(range(1, 32))
ticks_personalizados = [1, 11, 21, 31]
labels_dias = [f"{d:02d}" for d in ticks_personalizados]

for i, (month, (start, end)) in enumerate(periods.items()):
    row = i // 3
    col = i % 3
    ax = axes[row, col]

    for model, (color, data) in models.items():
        subplot = data.loc[start:end]
        if not subplot.empty:
            values = subplot.values
            ax.plot(
                dias_do_mes[: len(values)],
                values,
                color=color,
                label=model if i == 0 else None,
            )
    ax.set_title(month, fontsize=14)
    # Só mostra ticks do eixo x na última linha
    if row == 3:
        ax.set_xticks(ticks_personalizados)
        ax.set_xticklabels(labels_dias, fontsize=12)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=12)

plt.tight_layout()

output_file10 = os.path.join(output_dir, "comparacao_realpred_por_mes.pdf")
plt.savefig(output_file10)
