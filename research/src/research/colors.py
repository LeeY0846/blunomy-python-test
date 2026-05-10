from matplotlib.pyplot import cm
import numpy as np

colors = cm.tab20(np.linspace(0, 1, 20))

def get_color(label: int):
    if label < 0:
        return np.array([0, 0, 0, 1])  # black
    return colors[label % len(colors)]