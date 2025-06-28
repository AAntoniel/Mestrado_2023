from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import numpy as np
import pandas as pd

dates = pd.date_range(start="2019-01-01", end="2019-12-31")

# test_sw1y = pd.read_csv("output_ytrue_yhat/results_lstm-2025-02-20-09-52-26.csv") # lstm SW1Y
# test_sw1y = pd.read_csv("output_ytrue_yhat/results_NP-2024-12-12-20-29-38.csv") # NP ew
# test_sw1y = pd.read_csv(
#     "output_ytrue_yhat/results_NP-2024-12-26-14-24-33.csv"
# )  # NP sw1y
# test_sw1y = pd.read_csv(
#     "output_ytrue_yhat/results_NP-2024-12-16-20-57-34.csv"
# )  # NP sw2y
test_sw1y = pd.read_csv(
    "output_ytrue_yhat/results_chronos-2024-10-24-13-12-42.csv"
)  # CR
test_sw1y = test_sw1y[730:]
test_sw1y.index = dates
test_sw1y.ytrue = test_sw1y.ytrue.apply(lambda x: float(x.strip("[]")))
test_sw1y.yhat = test_sw1y.yhat.apply(lambda x: float(x.strip("[]")))

scoringSW1Y = dict(rmse=[], mae=[], mape=[])
for i in range(52):
    if i == 51:
        num_days = 8
    else:
        num_days = 7

    # Calcula os índices para os últimos dias
    start_idx = i * 7
    end_idx = start_idx + num_days

    # Obtém os últimos 7 ou 8 elementos de ytrue e yhat
    ytrue_last = test_sw1y.ytrue[start_idx:end_idx]
    yhat_last = test_sw1y.yhat[start_idx:end_idx]

    # print('ytrue_last', ytrue_last)
    # print('yhat_last', yhat_last)

    # Métricas para os últimos dias
    rmse = mean_squared_error(ytrue_last, yhat_last, squared=False)
    mae = mean_absolute_error(ytrue_last, yhat_last)
    mape = mean_absolute_percentage_error(ytrue_last, yhat_last)

    scoringSW1Y["rmse"].append(rmse)
    scoringSW1Y["mae"].append(mae)
    scoringSW1Y["mape"].append(mape)

# Média e desvio padrão das métricas
rmse_mean = round(np.mean(scoringSW1Y["rmse"]), 2)
rmse_std = round(np.std(scoringSW1Y["rmse"]), 2)

mae_mean = round(np.mean(scoringSW1Y["mae"]), 2)
mae_std = round(np.std(scoringSW1Y["mae"]), 2)

mape_mean = round((np.mean(scoringSW1Y["mape"]) * 100), 2)
mape_std = round((np.std(scoringSW1Y["mape"]) * 100), 2)

print(f"RMSE - Média: {rmse_mean}, Desvio Padrão: {rmse_std}")
print(f"MAE - Média: {mae_mean}, Desvio Padrão: {mae_std}")
print(f"MAPE - Média: {mape_mean}, Desvio Padrão: {mape_std}")
