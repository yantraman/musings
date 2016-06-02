import numpy as np

with open('data/german-credit/german_credit.csv') as f:
    data = np.array(map(lambda line: map(int, line.strip('\n').split(',')), f.readlines()))
data[:, -1] -= 1

np.random.shuffle(data)
nb_train = int(len(data) * 0.8)
X_train = data[:nb_train, :-1]
y_train = data[:nb_train, -1]
X_test = data[nb_train:, :-1]
y_test = data[nb_train:, -1]

assert len(X_train) == len(y_train) and (len(X_test) == len(y_test))

from sklearn import linear_model

model_lg = linear_model.LogisticRegression()  # specify the class weights: class_weight = {0:0.2, 1:0.8}
model_lg.fit(X_train, y_train)
y_test_pred = model_lg.predict(X_test)

print sum(y_test_pred == y_test) / float(len(y_test))
