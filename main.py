import pandas as pd
import weights_of_evidence_and_information_value
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from codificare_function import codificare
from analiza_model_function import analiza_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from afisare_rezultate_function import afisare_rezultate


def woe_and_iv_processing():
    airbnb_data = pd.read_csv("datasets/airbnb_dataset.csv", index_col=0)
    variables = list(airbnb_data)  # prima linie din csv

    airbnb_superhost = variables[4]  # a sasea coloana, variabila tinta

    variables.remove(airbnb_superhost)  # elimin variabila super_host
    predictori = variables

    inf_v, woe = weights_of_evidence_and_information_value.calculate(
        airbnb_data, predictori, airbnb_superhost)

    inf_v.to_csv("output/information_value.csv")
    woe.to_csv("output/weights_of_evidence.csv")

    print(inf_v)
    print("---")
    print(woe)


def classification_processing():
    airbnb_learning_data = pd.read_csv(
        "datasets/airbnb_dataset.csv", index_col=0)
    codificare(airbnb_learning_data)

    airbnb_applied_data = pd.read_csv(
        "datasets/airbnb_dataset_apply.csv")
    codificare(airbnb_applied_data)

    variables = list(airbnb_learning_data)
    airbnb_superhost = variables[4]  # a sasea coloana, variabila tinta
    variables.remove(airbnb_superhost)  # elimin variabila super_host
    predictori = variables

    x_invatare, x_testare, y_invatare, y_testare = \
        train_test_split(
            airbnb_learning_data[predictori],
            airbnb_learning_data[airbnb_superhost],
            test_size=0.4)

    # Fisher
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_testare, y_testare)
    z = lda.transform(x_testare)

    tabel_predictii = pd.DataFrame(index=airbnb_applied_data.index)
    tabel_predictii_testare = pd.DataFrame(
        {airbnb_superhost: y_testare}, index=x_testare.index)

    # Naive Bayes
    model_nb = GaussianNB()
    rezultate_bayes = analiza_model(
        model_nb, x_invatare, y_invatare, x_testare,
        y_testare, airbnb_applied_data, "naive_bayes", z,
        tabel_predictii_testare, tabel_predictii)

    afisare_rezultate(rezultate_bayes, "naive_bayes")

    # DecisionTree
    model_dt = DecisionTreeClassifier()
    rezultate_dt = analiza_model(
        model_dt, x_invatare, y_invatare, x_testare,
        y_testare, airbnb_applied_data, "decision_tree", z,
        tabel_predictii_testare, tabel_predictii)

    afisare_rezultate(rezultate_dt, "decision_tree")

    # knn
    model_knn = KNeighborsClassifier()
    rezultate_knn = analiza_model(
        model_knn, x_invatare, y_invatare, x_testare,
        y_testare, airbnb_applied_data, "knn", z,
        tabel_predictii_testare, tabel_predictii)

    afisare_rezultate(rezultate_knn, "knn")

    # Random Forest
    model_rf = RandomForestClassifier()
    rezultate_rf = analiza_model(
        model_rf, x_invatare, y_invatare, x_testare,
        y_testare, airbnb_applied_data, "random_forest", z,
        tabel_predictii_testare, tabel_predictii)
    afisare_rezultate(rezultate_rf, "random_forest")

    # SVM
    model_svm = SVC(probability=True)
    rezultate_svm = analiza_model(
        model_svm, x_invatare, y_invatare, x_testare,
        y_testare, airbnb_applied_data, "svm", z,
        tabel_predictii_testare, tabel_predictii)
    afisare_rezultate(rezultate_svm, "svm")

    # predictions
    tabel_predictii_testare.to_csv("output/predictii_test.csv")
    tabel_predictii.to_csv("output/predictii.csv")


if __name__ == "__main__":
    woe_and_iv_processing()
    classification_processing()
