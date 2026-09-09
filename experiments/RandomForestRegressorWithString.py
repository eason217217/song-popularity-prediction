import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;

df = pd.read_csv("../data/processed_tracks_new.csv");

x = df.drop(columns = ["popularity"])#, "time_signature", "explicit", "mode", "key", "instrumentalness", "liveness", "track_name", "tempo", "energy", "loudness", "speechiness", "valence", "danceability", "duration_ms"]);
y = df["popularity"];
df.drop_duplicates(inplace = True);
df.dropna(inplace = True);

cnt1 = 0
cnt2 = 0
cnt3 = 0;

for i in y.index:
    if (y[i] > 66.66):
        #y.drop(index = i, inplace = True);
        #x.drop(index = i, inplace = True);
        cnt1 += 1;
    elif (y[i] > 33.33 and cnt2 < 40000):
        y.drop(index = i, inplace = True);
        x.drop(index = i, inplace = True);
        cnt2 += 1
    elif(cnt3 < 40000):
        y.drop(index=i, inplace=True);
        x.drop(index=i, inplace=True);
        cnt3 += 1;

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error
from sklearn.tree import plot_tree

train_sizes = [1, 1000, 2836, 5671, 14, 8342, 10000]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42);
rfc = RandomForestRegressor(n_estimators = 50, random_state = 42);
rfc.fit(xTrain, yTrain);
#train_sizes, train_scores, validation_scores = learning_curve(
#rfc,
#X = x,
#y = y, train_sizes = train_sizes, cv = 5,
#scoring = 'neg_mean_squared_error')
#train_scores_mean = -train_scores.mean(axis = 1)
#validation_scores_mean = -validation_scores.mean(axis = 1)

#plt.plot(train_sizes, train_scores_mean, label = 'Training error')
#plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
#plt.ylabel('MSE', fontsize = 14)
#plt.xlabel('Training set size', fontsize = 14)
#plt.title('Learning curves for Random Forest Regressor', fontsize = 18, y = 1.03)
#plt.legend()
#plt.ylim(0,40)
#plt.autoscale();

yPred = rfc.predict(xTest);
print("MSE:", mean_squared_error(yTest, yPred), "R^2:", r2_score(yTest, yPred));

#importances = rfc.feature_importances_;
#sortedImportances = [];
#featuresNames = x.columns.values.tolist();

#for i in range(len(importances)):
#    temp = [];
#    temp.append(importances[i]);
#    temp.append(featuresNames[i]);
#    sortedImportances.append(temp);
#sortedImportances = sorted(sortedImportances);

#for i in range(len(importances)):
#    importances[i] = sortedImportances[i][0];
#    featuresNames[i] = sortedImportances[i][1];

#plt.barh(range(x.shape[1]), importances);
#plt.yticks(range(x.shape[1]), featuresNames);
#plt.title("Feature Importance in Spotify Musics");

plt.show();

plt.scatter(yTest, yPred);

plt.show();