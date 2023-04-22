import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from plot_functions import *


def analiza_model(model, x, y, x_, y_, x_app, nume_model, z,
                  t_predict_test, t_predict):
    model.fit(x, y)

    predictie = model.predict(x_)

    t_predict_test[nume_model] = predictie
    clase = model.classes_
    proba = model.predict_proba(x_)
    mat_c = confusion_matrix(y_, predictie)
    t_conf = pd.DataFrame(mat_c, clase, clase)
    t_conf["Acuratete"] = np.diag(mat_c) * 100 / np.sum(mat_c, axis=1)
    t_conf.to_csv("output/matc_" + nume_model + ".csv")
    acuratete = (
        sum(np.diag(mat_c)) * 100 / len(x_),
        t_conf["Acuratete"].mean(),
        cohen_kappa_score(y_, predictie)
    )
    t_acuratete = pd.Series(
        acuratete, ["Acuratete globala", "Acuratete medie", "Index Cohen"])
    t_acuratete.name = "Acuratete " + nume_model
    t_acuratete.to_csv("output/acuratete_" + nume_model + ".csv")
    if nume_model == "naive_bayes":
        plot_distributii(z, predictie, clase, nume_model)
    plot_cm(y_, predictie, nume_model)
    plot_metrici(y_, proba, nume_model)
    show()
    # Predictie in setul de aplicare
    predictie_ = model.predict(x_app)
    t_predict[nume_model] = predictie_
    return t_conf, t_acuratete


