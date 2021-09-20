import matplotlib.pyplot as plt
import numpy as np


def save_image(mel1, mel2, name):
    mel = np.concatenate([mel1, mel2], 1)
    plt.imshow(mel.T)
    plt.show()
    plt.savefig(name)
