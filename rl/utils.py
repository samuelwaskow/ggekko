import matplotlib.pyplot as plt
import numpy as np


def plot_daytrade(scores, filename, save_model=50):

    fig=plt.figure()
    ax1=fig.add_subplot(111, label="1")

    N = len(scores)
    x = [i+1 for i in range(N)]

    avg_scores = np.empty(N)
    
    for t in range(N):
        avg_scores[t] = np.mean(scores[max(0, t-save_model):(t+1)])

    ax1.plot(x, avg_scores)
    ax1.set(xlabel='', ylabel='Balance')

    fig.tight_layout()
    fig.savefig(filename)
    plt.close('all')