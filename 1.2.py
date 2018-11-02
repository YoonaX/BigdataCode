# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
data = [-35, 10, 21, 30, 40, 50, 60, 122, 300]
flierprops = {'marker': 'o', 'markerfacecolor': 'yellow', 'color': 'black'}
# plt.grid(True, linestyle = "-.", color = "black")
plt.boxplot(data,notch=False,flierprops=flierprops)
plt.savefig('D:/pythonprogram/BigData/picture/1.2.png')
plt.show()