import matplotlib.pyplot as plt
from seaborn import kdeplot
from scikitplot.metrics  \
    import plot_confusion_matrix, plot_roc, plot_cumulative_gain


def plot_metrici(y, proba, nume):
    fig = plt.figure(figsize=(13, 7))
    ax = fig.subplots(1, 2, sharey=True)
    plot_roc(y, proba, "Graficul Roc. Metoda:" + nume, ax=ax[0])
    plot_cumulative_gain(y, proba, "Graficul Gain. Metoda:" + nume, ax=ax[1])
    plt.savefig("output/ROC_Gain_" + nume + ".png")


def plot_cm(y, predictie, nume_model):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    plot_confusion_matrix(y, predictie, normalize=True, ax=ax,
                          title="Matrice Confuzie. Model:" + nume_model)
    plt.savefig("output/MatC_" + nume_model + ".png")


def plot_distributii(z, y, clase, nume_model):
    fig = plt.figure(figsize=(10, 7))
    q = len(clase) - 1
    if q == 1:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(
            "Functii Fisher. Model:" + nume_model, fontsize=16, color='b')
        kdeplot(x=z[:, 0], hue=y, hue_order=clase, fill=True, ax=ax)
    else:
        fig.suptitle(
            "Functii Fisher. Model:" + nume_model, fontsize=16, color='b')
        ax = fig.subplots(q, 1, sharex=True)
        for i in range(q):
            kdeplot(x=z[:, i], hue=y, hue_order=clase, fill=True, ax=ax[i])
    plt.savefig("output/Distrib_" + nume_model + ".png")


def show():
    plt.show()
