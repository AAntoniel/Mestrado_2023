#  Calculo num. épocas
import math

T = 1095

n_epochs = (1000 * (2 ** (5 / 2 * math.log10(T)))) / T

print(n_epochs)
