import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;

df = pd.read_csv("../data/dataset.csv");
df.drop(df.columns[[0]], axis = 1, inplace = True);
df.dropna(inplace = True);
#print(df);
#print(df.info())

cnt = 0;
for bool in df.duplicated():
    if (bool):
        cnt = cnt + 1;
print("Duplicates:", cnt);

df.drop_duplicates(inplace = True);
numDf = df.select_dtypes(include = [np.number]);
#sns.heatmap(numDf.corr().round(3), center = 0, annot = True);
#plt.show();

x = numDf.drop(columns = ["popularity"]);
#y = numDf["popularity"];

#for i in y.index:
#    if (y[i] >= 50):
#        y.update(pd.Series([1], index=[i]));
#    else:
#        y.update(pd.Series([0], index=[i]));
#y.to_csv("y2.csv", index = False);

#for i in y.index:
#    if (y[i] >= 75):
#        y.update(pd.Series([4], index=[i]));
#    elif (y[i] >= 50):
#        y.update(pd.Series([3], index=[i]));
#    elif (y[i] >= 25):
#        y.update(pd.Series([2], index=[i]));
#    else:
#        y.update(pd.Series([1], index=[i]));

#y.to_csv("y4.csv", index = False);

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, ConfusionMatrixDisplay

y = pd.read_csv("../data/y2.csv");
y = y["popularity"];

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2);
rfc = RandomForestClassifier();
rfc.fit(xTrain, yTrain);

yPred = rfc.predict(xTest);
print("Accuracy Score:", accuracy_score(yTest, yPred));

cm = confusion_matrix(yTest, yPred);
ConfusionMatrixDisplay(confusion_matrix=cm).plot();

#rfc_score = cross_val_score(rfc, x, y, cv = 10);
#plt.plot(range(1, 11), rfc_score);

plt.show();
