# -*- coding:  utf-8 -*-

import matplotlib.pyplot as plt

line1, = plt.plot([1, 2, 3], linestyle='--')
line2, = plt.plot([3, 2, 1], lineWidth=4)

# Create a legend for the first line.
plt.legend([line1, line2], ["Line1", "Line2"], loc=1)
plt.show()