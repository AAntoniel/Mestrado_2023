# https://kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import datetime
import os

from plot import setup, wrapup, save

matplotlib.use("TkAgg")
# Obter a data e hora atuais
current_time = datetime.datetime.now()

# Formatar a data e hora no formato desejado
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# Utilizar o estilo definido em plot.py
# plt.rcParams.update(plt.rcParamsDefault)
# plt.style.use("seaborn-whitegrid")
# palette = plt.get_cmap("tab10")


class SSA(object):

    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.

        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list.
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.

        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """

        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError(
                "Unsupported time series object. Try Pandas Series, NumPy array or list."
            )

        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1

        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i : L + i] for i in range(0, self.K)]).T

        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)

        self.TS_comps = np.zeros((self.N, self.d))

        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array(
                [
                    self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                    for i in range(self.d)
                ]
            )

            # Diagonally average the elementary matrices, store them as columns in array.
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:, i] = [
                    X_rev.diagonal(j).mean()
                    for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])
                ]

            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                X_rev = X_elem[::-1]
                self.TS_comps[:, i] = [
                    X_rev.diagonal(j).mean()
                    for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])
                ]

            self.X_elem = (
                "Re-run with save_mem=False to retain the elementary matrices."
            )

            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."

        # Calculate the w-correlation matrix.
        self.calc_wcorr()

    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d

        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(
            self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index
        )

    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.

        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int):
            indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """

        # Calculate the weights
        w = np.array(
            list(np.arange(self.L) + 1)
            + [self.L] * (self.K - self.L - 1)
            + list(np.arange(self.L) + 1)[::-1]
        )

        def w_inner(F_i, F_j):
            return w.dot(F_i * F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array(
            [w_inner(self.TS_comps[:, i], self.TS_comps[:, i]) for i in range(self.d)]
        )
        F_wnorms = F_wnorms**-0.5

        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.Wcorr[i, j] = abs(
                    w_inner(self.TS_comps[:, i], self.TS_comps[:, j])
                    * F_wnorms[i]
                    * F_wnorms[j]
                )
                self.Wcorr[j, i] = self.Wcorr[i, j]

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d

        if self.Wcorr is None:
            self.calc_wcorr()

        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d - 1
        else:
            max_rnge = max

        plt.xlim(min - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min - 0.5)
        # plt.show()


# Criação de um diretório para as decomposições, se não exisistir
output_dir1 = "output/analise_ssa/analise_ssa2y"
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)

# Criação de um diretório para as decomposições, se não exisistir
output_dir2 = "output/analise_ssa/analise_ssa3y"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

df = pd.read_csv("data/dataset.csv")
df = df.dropna()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df = pd.DataFrame(df)

# -------------------------------------------------------------------------------
# ANÁLISE COM DADOS DE TREINAMENTO (2016 E 2017)
# -------------------------------------------------------------------------------

# w-corr matrix
# df_ssa_l2 = SSA(df["water_produced"][:730], 274)
# df_ssa_l2.plot_wcorr(max=60)
# plt.grid(False)
# plt.title(r"W-Correlation para dados de treinamento")
# plt.tight_layout()
# output_path = os.path.join(output_dir1, "wcorrelation_2y.pdf")
# plt.savefig(output_path, format="pdf")
# plt.clf()

# # reconstrução dos componentes relevantes
# componentes_2y = [
#     df_ssa_l2.reconstruct(0),
#     df_ssa_l2.reconstruct([1, 2]),
#     df_ssa_l2.reconstruct([3, 4]),
#     df_ssa_l2.reconstruct(slice(5, 10)),
#     df_ssa_l2.reconstruct([10, 11, 12]),
#     df_ssa_l2.reconstruct(slice(13, 18)),
#     df_ssa_l2.reconstruct([18, 19]),
# ]

# # plot de todos os componentes agrupados
# plt.figure(figsize=(11.69, 8.27))
# colors = ["b", "orange", "green", "red", "purple", "brown", "#ABABAB"]
# df_ssa_l2.orig_TS.plot(alpha=0.4, color="green")
# for i, componente in enumerate(componentes_2y):
#     componente.plot(color=colors[i])
# plt.xlabel("$t$")
# plt.ylabel(r"$\tilde{F}_i(t)$")
# plt.title("Componentes Agrupados, $L=274$")
# legend =  ["Série Original"] + [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(7)]
# plt.legend(legend)
# plt.tight_layout()
# output_path = os.path.join(output_dir1, "componentes_agrupados_2y.pdf")
# plt.savefig(output_path, format="pdf")
# plt.clf()

# # plot de todos os componentes separadamente
# fig, axes = plt.subplots(4, 2, figsize=(8.27, 11.69), sharex=True)
# axes = axes.flatten()
# start_2017 = "2017-01-01"
# end_2017 = "2017-12-31"
# ylim = (-5000, 25000)

# df_ssa_l2.orig_TS[start_2017:end_2017].plot(
#     ax=axes[0], title="Série Original", alpha=0.4, color="green", ylim=ylim
# )
# for i, componente in enumerate(componentes_2y):
#     componente[start_2017:end_2017].plot(
#         ax=axes[i + 1],
#         title=r"Componente $\tilde{F}^{(%d)}$" % i,
#         color=colors[i],
#         ylim=ylim,
#     )

# plt.tight_layout()
# output_path = os.path.join(output_dir1, "componentes_separados_2y.pdf")
# plt.savefig(output_path, format="pdf")
# plt.clf()
# # plt.show()

# # plot dos componentes relevantes agrupados x o restante dos componentes
# plt.figure(figsize=(11.69, 8.27))
# df_ssa_l2.orig_TS.plot(alpha=0.4, color="green")
# df_ssa_l2.reconstruct(slice(0, 20)).plot(color=colors[0])
# df_ssa_l2.reconstruct(slice(20, 730)).plot(color=colors[1])
# plt.legend(
#     ["Série original", "Primeiros 20 componentes juntos", "730 Componentes restantes"]
# )
# plt.tight_layout()
# output_path = os.path.join(output_dir1, "combinacao_componentes_2y.pdf")
# plt.savefig(output_path, format="pdf")
# plt.clf()
# plt.show()

# -------------------------------------------------------------------------------
# ANÁLISE COM DADOS DE TREINAMENTO E VALIDAÇÃO (2016, 2017 E 2018)
# -------------------------------------------------------------------------------

# w-corr matrix
df_ssa_l3 = SSA(df["water_produced"][:1095], 410)
df_ssa_l3.plot_wcorr(max=60)
plt.grid(False)
plt.title(r"W-Correlation incluindo dados de validação")
plt.tight_layout()
output_path = os.path.join(output_dir2, "wcorrelation_3y.pdf")
plt.savefig(output_path, format="pdf")
plt.clf()
# plt.show()

# reconstrução dos componentes relevantes
componentes_3y = [
    df_ssa_l3.reconstruct(0),
    df_ssa_l3.reconstruct([1, 2]),
    df_ssa_l3.reconstruct([3, 4]),
    df_ssa_l3.reconstruct([5, 6]),
    df_ssa_l3.reconstruct([7, 8]),
    df_ssa_l3.reconstruct(slice(9, 13)),
    df_ssa_l3.reconstruct([13, 14]),
    df_ssa_l3.reconstruct(slice(15, 21)),
]

# plot de todos os componentes agrupados
plt.figure(figsize=(11.69, 8.27))
colors = ["b", "orange", "green", "red", "purple", "brown", "#ABABAB", "#f781bf"]
df_ssa_l3.orig_TS.plot(alpha=0.4, color="green")
for i, componente in enumerate(componentes_3y):
    componente.plot(color=colors[i])
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.title("Componentes Agrupados, $L=274$")
legend = ["Série Original"] + [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(8)]
plt.legend(legend)
plt.tight_layout()
output_path = os.path.join(output_dir2, "componentes_agrupados_3y.pdf")
plt.savefig(output_path, format="pdf")
plt.clf()
# plt.show()

# plot de todos os componentes separadamente
fig, axes = plt.subplots(4, 2, figsize=(8.27, 11.69), sharex=True)
axes = axes.flatten()
start_2018 = "2018-01-01"
end_2018 = "2018-12-31"
ylim = (-5000, 25000)

df_ssa_l3.orig_TS[start_2018:end_2018].plot(
    ax=axes[0], title="Série Original", alpha=0.4, color="green", ylim=ylim
)
componentes_3y[0][start_2018:end_2018].plot(ax=axes[0], color=colors[0], ylim=ylim)
for i, componente in enumerate(componentes_3y[1:], start=1):
    componente[start_2018:end_2018].plot(
        ax=axes[i],
        title=r"Componente $\tilde{F}^{(%d)}$" % (i),
        color=colors[i],
        ylim=ylim,
    )

plt.tight_layout()
output_path = os.path.join(output_dir2, "componentes_separados_3y.pdf")
plt.savefig(output_path, format="pdf")
plt.clf()
# plt.show()

# plot dos componentes relevantes agrupados x o restante dos componentes
plt.figure(figsize=(11.69, 8.27))
df_ssa_l3.orig_TS.plot(alpha=0.4, color="green")
df_ssa_l3.reconstruct(slice(0, 21)).plot(color=colors[0])
df_ssa_l3.reconstruct(slice(21, 1095)).plot(color=colors[1])
plt.legend(
    ["Série original", "Primeiros 21 componentes juntos", "1074 Componentes restantes"]
)
plt.tight_layout()
output_path = os.path.join(output_dir2, "combinacao_componentes_3y.pdf")
plt.savefig(output_path, format="pdf")
plt.clf()
