
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from neural import MLP



df = pd.read_csv('data.csv')

# возьмем перые 100 строк, 4-й столбец
y = df.iloc[0:200, 4].values
# так как ответы у нас строки - нужно перейти к численным значениям

y = pd.get_dummies(y).values

print(y.shape)

X = df.iloc[0:200, :4].values

# добавим фиктивный признак для удобства матричных вычслений
X = np.concatenate([np.ones((len(X),1)), X], axis = 1)

print(X.shape)


# инициализируем нейронную сеть
inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи
hiddenSizes = 10 # задаем число нейронов скрытого слоя
outputSize = y.shape[1]  # количество выходных сигналов равно количеству классов задачи


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

iterations = 50
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)


# обучаем сеть (фактически сеть это вектор весов weights)
for i in range(iterations):
    net.train(X_train, y_train)
    #for j in range(150):
    #    net.train(np.array([X[j]]), np.array([y[j]]))

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - net.predict(X)))))

# считаем ошибку на обучающей выборке
pr = net.predict(X)
print(sum(abs(y-(pr>0.5))))


lr_auc = roc_auc_score(y_test, net.predict(X_test))

print(lr_auc)


v = 0
c = 0
for a, b in zip(np.round(net.predict(X_test)), y_test):
    #print([list(a), list(b), list(a)==list(b)])
    if list(a)==list(b):
        v += 1
    c += 1
print("Acc", v/c* 100, "%")