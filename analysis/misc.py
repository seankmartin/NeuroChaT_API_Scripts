import matplotlib.pyplot as plt
import numpy as np


def visualise_spacing(N=61, start=5, stop=10000):
    """Plots a visual representation logspacing"""
    # This is equivalent to np.exp(np.linspace)
    x1 = np.logspace(np.log10(start), np.log10(stop), N, base=10)
    y = np.zeros(N)
    plt.plot(x1, y, 'o')
    plt.ylim([-0.5, 1])
    print(x1)
    plt.show()
