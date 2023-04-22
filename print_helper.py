import pandas as pd

#print(pd.read_csv("output/predictii_test.csv", index_col=0))

predictii = pd.read_csv("output/predictii.csv", index_col=0)
# print(predictii)

df = pd.DataFrame(predictii)
print(df)

numar_predictii_eronate = len(df[df["random_forest"] == False])
print("\nPrevizionate gresit:")
print(numar_predictii_eronate)

numar_predictii_corecte = len(df[df["random_forest"] == True])
print("\numar_predictii_corecte corect:")
print(numar_predictii_corecte)
