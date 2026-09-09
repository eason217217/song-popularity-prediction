import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;

df = pd.read_csv("../data/processed_tracks_new.csv");
df.dropna(inplace = True);
df.drop_duplicates(inplace = True);

print(df.info());

x = df.drop(columns = ["popularity"]);
y = pd.read_csv("../data/y4.csv")["popularity"];

#for i in y.index:
#    if (y[i] >= 50):
#        y.update(pd.Series([2], index=[i]));
#    else:
#        y.update(pd.Series([1], index=[i]));
#y.to_csv("y3.csv", index = False);

#for i in y.index:
#    if (y[i] >= 66.66):
#        y.update(pd.Series([3], index=[i]));
#    elif (y[i] >= 33.33):
#        y.update(pd.Series([2], index=[i]));
#    else:
#        y.update(pd.Series([1], index=[i]));
#y.to_csv("y3.csv", index = False);

#for i in y.index:
#    if (y[i] >= 75):
#        y.update(pd.Series([4], index=[i]));
#    elif (y[i] >= 50):
#        y.update(pd.Series([3], index=[i]));
#    elif (y[i] >= 25):
#        y.update(pd.Series([2], index=[i]));
#    else:
#        y.update(pd.Series([1], index=[i]));
#
#y.to_csv("y4.csv", index = False);

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, ConfusionMatrixDisplay

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42);
rfc = RandomForestClassifier(n_estimators = 4, random_state = 42);
rfc.fit(xTrain, yTrain);

yPred = rfc.predict(xTest);
print("Accuracy Score:", accuracy_score(yTest, yPred));


cm = confusion_matrix(yTest, yPred);
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["0-49", "50-100", "a", "b"]).plot();
plt.show();
