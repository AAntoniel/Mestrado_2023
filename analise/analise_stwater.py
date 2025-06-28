import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools
import datetime
import os

from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from plot import setup, wrapup, save

matplotlib.use("TkAgg")

# Data e hora atuais
current_time = datetime.datetime.now()

# Formatação data e hora
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# Diretórios
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv("data/dataset.csv")
df = df.dropna()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Utilizar o estilo definido em plot.py
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use("seaborn-whitegrid")
palette = plt.get_cmap("tab10")

font_size_title = 20
font_size_labels = 16
font_size_ticks = 14

# plt.figure(figsize=(18, 8))
# plt.plot(
#     df["water_produced"][1095:],
#     color=palette(0),
# )
# # plt.title("Water produced at the water treatment plants (WTPs) from 2016 to 2019")
# plt.xlabel("Período", fontsize=14)
# plt.ylabel("Produção de água m³", fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend()
# plt.show()

# Salvar o gráfico em PDF
# save(None, os.path.join(output_dir, f"water_produced2.pdf"))

# Decomposição da série temporal por modelo multiplicativo
result = seasonal_decompose(
    df["water_produced"][:1095], model="multiplicative", period=365
)

# Componentes individuais
trend_component = result.trend
seasonal_component = result.seasonal
residual_component = result.resid
x_min = df.index[:1095].min()
x_max = df.index[:1095].max()

# subplots
fig, axs = plt.subplots(4, 1, figsize=(16, 12))

axs[0].plot(df["water_produced"][:1095])
axs[0].set_title("Série Temporal", fontsize=font_size_title)
axs[0].set_xlabel("Ano", fontsize=font_size_labels)
axs[0].set_ylabel("Água produzida (m³)", fontsize=font_size_labels)
axs[0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[0].set_xlim(x_min, x_max)

axs[1].plot(trend_component)
axs[1].set_title("Tendência", fontsize=font_size_title)
axs[1].set_xlabel("Ano", fontsize=font_size_labels)
axs[1].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[1].set_xlim(x_min, x_max)

axs[2].plot(seasonal_component)
axs[2].set_title("Sazonalidade", fontsize=font_size_title)
axs[2].set_xlabel("Ano", fontsize=font_size_labels)
axs[2].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[2].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[2].set_xlim(x_min, x_max)

axs[3].plot(residual_component)
axs[3].set_title("Ruído", fontsize=font_size_title)
axs[3].set_xlabel("Ano", fontsize=font_size_labels)
axs[3].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[3].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[3].set_xlim(x_min, x_max)

plt.tight_layout()
# plt.show()

save(None, os.path.join(output_dir, f"multiplicative_decomposition_val.pdf"))

# Decomposição da série temporal por modelo aditivo
result = seasonal_decompose(df["water_produced"][:1095], model="Aditive", period=365)

trend_component = result.trend
seasonal_component = result.seasonal
residual_component = result.resid

plt.plot(result.resid)
plt.show()

# subplots
fig, axs = plt.subplots(4, 1, figsize=(16, 12))

axs[0].plot(df["water_produced"][:1095])
axs[0].set_title("Série Temporal", fontsize=font_size_title)
axs[0].set_xlabel("Ano", fontsize=font_size_labels)
axs[0].set_ylabel("Água produzida (m³)", fontsize=font_size_labels)
axs[0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[0].set_xlim(x_min, x_max)

axs[1].plot(trend_component)
axs[1].set_title("Tendência", fontsize=font_size_title)
axs[1].set_xlabel("Ano", fontsize=font_size_labels)
axs[1].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[1].set_xlim(x_min, x_max)

axs[2].plot(seasonal_component)
axs[2].set_title("Sazonalidade", fontsize=font_size_title)
axs[2].set_xlabel("Ano", fontsize=font_size_labels)
axs[2].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[2].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[2].set_xlim(x_min, x_max)

axs[3].plot(residual_component)
axs[3].set_title("Ruído", fontsize=font_size_title)
axs[3].set_xlabel("Ano", fontsize=font_size_labels)
axs[3].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[3].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[3].set_xlim(x_min, x_max)


plt.tight_layout()
# plt.show()

save(None, os.path.join(output_dir, f"aditive_decomposition_val.pdf"))

# # ---------------------- DECOMPOSIÇÃO COM DADOS DE TREINAMENTO ------------------------------

x_min2 = df.index[:730].min()
x_max2 = df.index[:730].max()
result = seasonal_decompose(
    df["water_produced"][:730], model="multiplicative", period=365
)

trend_component = result.trend
seasonal_component = result.seasonal
residual_component = result.resid

# subplots
fig, axs = plt.subplots(4, 1, figsize=(16, 12))

font_size_title = 20
font_size_labels = 16
font_size_ticks = 14

axs[0].plot(df["water_produced"][:730])
axs[0].set_title("Série Temporal", fontsize=font_size_title)
axs[0].set_xlabel("Ano", fontsize=font_size_labels)
axs[0].set_ylabel("Água produzida (m³)", fontsize=font_size_labels)
axs[0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[0].set_xlim(x_min2, x_max2)

axs[1].plot(trend_component)
axs[1].set_title("Tendência", fontsize=font_size_title)
axs[1].set_xlabel("Ano", fontsize=font_size_labels)
axs[1].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[1].set_xlim(x_min2, x_max2)

axs[2].plot(seasonal_component)
axs[2].set_title("Sazonalidade", fontsize=font_size_title)
axs[2].set_xlabel("Ano", fontsize=font_size_labels)
axs[2].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[2].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[2].set_xlim(x_min2, x_max2)

axs[3].plot(residual_component)
axs[3].set_title("Ruído", fontsize=font_size_title)
axs[3].set_xlabel("Ano", fontsize=font_size_labels)
axs[3].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[3].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[3].set_xlim(x_min2, x_max2)

plt.tight_layout()
# plt.show()

save(None, os.path.join(output_dir, f"multiplicative_decomposition_train.pdf"))

# Decomposição da série temporal por modelo multiplicativo
result = seasonal_decompose(df["water_produced"][:730], model="Aditive", period=365)

trend_component = result.trend
seasonal_component = result.seasonal
residual_component = result.resid

# subplots
fig, axs = plt.subplots(4, 1, figsize=(16, 12))

axs[0].plot(df["water_produced"][:730])
axs[0].set_title("Série Temporal", fontsize=font_size_title)
axs[0].set_xlabel("Ano", fontsize=font_size_labels)
axs[0].set_ylabel("Água produzida (m³)", fontsize=font_size_labels)
axs[0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[0].set_xlim(x_min2, x_max2)

axs[1].plot(trend_component)
axs[1].set_title("Tendência", fontsize=font_size_title)
axs[1].set_xlabel("Ano", fontsize=font_size_labels)
axs[1].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[1].set_xlim(x_min2, x_max2)

axs[2].plot(seasonal_component)
axs[2].set_title("Sazonalidade", fontsize=font_size_title)
axs[2].set_xlabel("Ano", fontsize=font_size_labels)
axs[2].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[2].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[2].set_xlim(x_min2, x_max2)

axs[3].plot(residual_component)
axs[3].set_title("Ruído", fontsize=font_size_title)
axs[3].set_xlabel("Ano", fontsize=font_size_labels)
axs[3].set_ylabel("Água produzida", fontsize=font_size_labels)
axs[3].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[3].set_xlim(x_min2, x_max2)


plt.tight_layout()
# plt.show()

save(None, os.path.join(output_dir, f"aditive_decomposition_train.pdf"))

# ------------------- NORMAS DOS RESÍDUOS ------------------------------

from sklearn.linear_model import LinearRegression

model = LinearRegression()

valores = df["water_produced"][:1095].values

intervalos_lag = [1, 7, 30, 365]
normas_residuo = []

fig, axs = plt.subplots(2, 2, figsize=(15, 8))
axs = axs.ravel()

for i, lag in enumerate(intervalos_lag):
    # Aplica os valores de lag as observações
    x = df["water_produced"][:1095].values[:-lag]
    y = df["water_produced"][:1095].values[lag:]

    model.fit(x.reshape(-1, 1), y)
    pred = model.predict(x.reshape(-1, 1))
    print("I", i, model.coef_)

    # Calcula a norma dos resíduos
    residuos = np.sqrt(np.sum((y - pred) ** 2))
    normas_residuo.append(residuos)

    axs[i].scatter(x, y)
    axs[i].set_xlabel("Zt", fontsize=font_size_labels)
    axs[i].set_ylabel(f"Zt + Lag ({lag})", fontsize=font_size_labels)
    axs[i].set_title(
        f"Lag {lag} - Norma de Resíduo: {normas_residuo[i]:.2f}",
        fontsize=font_size_title,
    )
    axs[i].plot(x, pred, color="r")
    axs[i].tick_params(axis="both", which="major", labelsize=font_size_ticks)

plt.tight_layout()
# plt.show()
save(None, os.path.join(output_dir, f"normas_residuo.pdf"))

# ------------ TESTES DE ESTACIONARIEDADE -------------------------------------


def teste_ADF(serie):
    # Teste de Dickey-Fuller Aumentado (ADF)
    resultado_adf = adfuller(serie)
    print("Teste de Dickey-Fuller Aumentado (ADF)")
    print(f"Estatística do teste: {resultado_adf[0]}")
    print(f"Valor-p: {resultado_adf[1]}")
    print(f"Número de lags utilizados: {resultado_adf[2]}")
    print(f"Número de observações utilizadas: {resultado_adf[3]}")
    print("Resultado do teste:")
    if resultado_adf[1] <= 0.05:
        print("A série é estacionária.")
    else:
        print("A série não é estacionária.")


teste_ADF(df.water_produced[:1095])


def teste_KPSS(serie):
    # Teste KPSS
    resultado_kpss = kpss(serie)
    print("Teste KPSS")
    print(f"Estatística do teste: {resultado_kpss[0]}")
    print(f"Valor-p: {resultado_kpss[1]}")
    print(f"Número de lags utilizados: {resultado_kpss[2]}")
    print(f"Número de observações utilizadas: {resultado_kpss[3]}")
    print("Resultado do teste:")
    if resultado_kpss[1] <= 0.05:
        print("A série não é estacionária.")
    else:
        print("A série é estacionária.")


teste_KPSS(df.water_produced[:1095])

# ------------------- ST COM DIFF --------------------------------------

# ST com diferenciação
plt.figure(figsize=(18, 8))
plt.plot(
    df["water_produced"].diff(),
    color=palette(0),
)
plt.title(
    "Quantidade de água produzida diariamente entre 2016 e 2019 (após técnica de diferenciação)",
    fontsize=font_size_title,
)
plt.xlabel("Period", fontsize=16)
plt.ylabel("Water produced (diff)³", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
# plt.show()

save(None, os.path.join(output_dir, f"water_produced_diff.pdf"))

# ------------------- ACF e PACF VALIDAÇÃO --------------------------------------
# subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot ACF no primeiro subplot
plot_acf(df.water_produced[:1095], ax=axs[0], lags=40)
axs[0].set_title("Autocorrelação (ACF)", fontsize=font_size_title)
axs[0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[0].set_xlabel("Lags", fontsize=font_size_labels)

# Plot PACF no segundo subplot
plot_pacf(df.water_produced[:1095], ax=axs[1])
axs[1].set_title("Autocorrelação Parcial (PACF)", fontsize=font_size_title)
axs[1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[1].set_xlabel("Lags", fontsize=font_size_labels)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ACF_PACF_val.pdf"))

# subplots
df_diff = df.water_produced[:1095].diff().dropna()
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot ACF no primeiro subplot
plot_acf(df_diff, ax=axs[0], lags=40)
axs[0].set_title("Autocorrelação (ACF)", fontsize=font_size_title)
axs[0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[0].set_xlabel("Lags", fontsize=font_size_labels)

# Plot PACF no segundo subplot
plot_pacf(df_diff, ax=axs[1])
axs[1].set_title("Autocorrelação Parcial (PACF)", fontsize=font_size_title)
axs[1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[1].set_xlabel("Lags", fontsize=font_size_labels)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ACF_PACF_val_(Diff).pdf"))

# ------------------- ACF e PACF TREINAMENTO --------------------------------------

# subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot ACF no primeiro subplot
plot_acf(df.water_produced[:730], ax=axs[0], lags=40)
axs[0].set_title("Autocorrelação (ACF)", fontsize=font_size_title)
axs[0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[0].set_xlabel("Lags", fontsize=font_size_labels)

# Plot PACF no segundo subplot
plot_pacf(df.water_produced[:730], ax=axs[1])
axs[1].set_title("Autocorrelação Parcial (PACF)", fontsize=font_size_title)
axs[1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[1].set_xlabel("Lags", fontsize=font_size_labels)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ACF_PACF_train.pdf"))

# subplots
df_diff = df.water_produced[:730].diff().dropna()
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot ACF no primeiro subplot
plot_acf(df_diff, ax=axs[0], lags=40)
axs[0].set_title("Autocorrelação (ACF)", fontsize=font_size_title)
axs[0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[0].set_xlabel("Lags", fontsize=font_size_labels)

# Plot PACF no segundo subplot
plot_pacf(df_diff, ax=axs[1])
axs[1].set_title("Autocorrelação Parcial (PACF)", fontsize=font_size_title)
axs[1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
axs[1].set_xlabel("Lags", fontsize=font_size_labels)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ACF_PACF_train_(Diff).pdf"))
