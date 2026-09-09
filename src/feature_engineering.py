import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading processed data...")
data = pd.read_csv('processed_tracks.csv')
print("Data loaded successfully.")


X = data.drop(columns=['popularity'])
y = data['popularity']
print("Features and target variable separated.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("Training and testing sets saved.")