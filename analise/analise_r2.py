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
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

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

"""#Separação dos dados por mês"""

# Separar os dados de ytrue e yhat por mês
# ------------------------------------------------------------------------------

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

# Times New Roman e aumentar o tamanho da fonte
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
dfs = [test_sw2y_arima, test_ew_sarima, test_sw2y_lstm, test_ew_np, test_sw2y_cr]
model_names = ["ARIMA-SW2Y", "SARIMA-EW", "LSTM-SW2Y", "NP-EW", "CR-SW2Y"]

fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))
axes = axes.flatten()

for i in range(4):
    ax = axes[i]
    dfpred = dfs[i]
    model_name = model_names[i]

    sns.regplot(
        x="ytrue",
        y="yhat",
        data=dfpred,
        ax=ax,
        scatter=True,
        label=model_name,
        scatter_kws={"alpha": 0.5},
    )

    r2 = round(r2_score(dfpred["ytrue"], dfpred["yhat"]), 3)
    print(f"{i}: ", r2)
    ax.set_title(f"{model_name}: $R^2$={r2}")

axes[4].axis("off")
axes[5].axis("off")

# Ajuste manual do subplot centralizado
left = 0.3  # Ajuste horizontal
bottom = 0.05  # Ajuste vertical
width = 0.41
height = 0.25

ax = fig.add_axes([left, bottom, width, height])  # Adiciona o eixo manualmente

# Plotar o último modelo
dfpred = dfs[4]
model_name = model_names[4]

sns.regplot(
    x="ytrue",
    y="yhat",
    data=dfpred,
    ax=ax,
    scatter=True,
    label=model_name,
    scatter_kws={"alpha": 0.5},
)

r2 = round(r2_score(dfpred["ytrue"], dfpred["yhat"]), 3)
print(r2)
ax.set_title(f"{model_name}: $R^2$={r2}")

# Ajustar todos os eixos
for ax in fig.axes:
    if ax.has_data():  # Só aplica em eixos com gráfico
        ax.yaxis.set_major_locator(MultipleLocator(5000))
        ax.xaxis.set_major_locator(MultipleLocator(5000))
        ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.xaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        ax.set(xlabel="Real", ylabel="Previsto")

plt.tight_layout(rect=[0, 0.1, 1, 1])

output_dir = "comparativos"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "realxpred_r2.pdf")
plt.savefig(output_file)
