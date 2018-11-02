# coding: utf-8
# data = [-35, 10, 21, 30, 40, 50, 60, 71, 126]
import matplotlib.pyplot as plt
data = [10, 20]
# data = [-35, 10]
print( (-35 + 10 + 21 + 30 + 40 + 50 + 60 + 21 + 126 + 71) / 10)
data = [-35, 10, 21, 30, 40, 50, 60, 21, 126, 71]
flierprops = {'marker': 'x', 'markerfacecolor': 'red', 'color': 'black'}
# plt.grid(True, linestyle = "-.", color = "black", widths = 2, meanline = None)
plt.boxplot(data, notch=False, flierprops=flierprops)
plt.show()
