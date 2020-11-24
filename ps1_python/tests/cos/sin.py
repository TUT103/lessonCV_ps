import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10.0, 10.0, 0.1)
plt.plot(x, np.sin(x))
plt.plot(x, x * 0)
plt.show()
