import torch
import numpy as np
import pandas as pd
from neural import MLPptorch2

import torch.nn as nn


class Perceptron(nn.Module):

    af_alias = {""}
    # как и раньше для инициализации на вход нужно подать размеры входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size, hidden_count=1, afuncs=[]):
        if not afuncs:
            afuncs = [nn.Sigmoid(), nn.Sigmoid()]
        assert len(afuncs) == hidden_count + 1
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет и позволяет запускать их одновременно
        gen_layers = [nn.Linear(in_size, hidden_size), afuncs[0]]

        for k in range(hidden_count):
            gen_layers += [nn.Linear(hidden_size, hidden_size), afuncs[k + 1]]
        gen_layers += [nn.Linear(hidden_size, out_size), afuncs[-1]]
        self.layers = nn.Sequential(*gen_layers)
    # прямой проход
    def forward(self,x):
        return self.layers(x)

# функция обучения
def train(net, x, y, num_iter, lossFn, optimizer):
    losses_values = []
    for i in range(0,num_iter):
        pred = net.forward(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        if i%100==0:
           l = loss.item()
           print('Ошибка на ' + str(i+1) + ' итерации: ', l)
        losses_values.append(l)
    return losses_values


# теперь можно использовать созданный класс на практике

df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)
X = df.iloc[0:100, 0:3].values


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи

net = MLPptorch2(inputSize, hiddenSizes, outputSize, 3)
net2 = MLPptorch2(inputSize, hiddenSizes, outputSize, 3, nn.Sigmoid)

res_loss = train(net, torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)),500, nn.MSELoss(), torch.optim.SGD(net.parameters(), lr=0.001))
res_loss2 = train(net2, torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)), 500, nn.MSELoss(), torch.optim.SGD(net2.parameters(), lr=0.001))

import matplotlib.pyplot as plt

with torch.no_grad():

    print(list(np.where(np.array(net.forward(torch.from_numpy(X.astype(np.float32)))) == 1, "Iris-setosa", "Not Iris-setosa")))
    print(list(np.where(np.array(net2.forward(torch.from_numpy(X.astype(np.float32)))) == 1, "Iris-setosa", "Not Iris-setosa")))

    print("Errors from 1 model", np.sum(np.abs(np.round(np.array(net.forward(torch.from_numpy(X.astype(np.float32))))) - np.round(y.astype(np.float32)))))
    print("Errors from 2 model", np.sum(np.abs(np.round(np.array(net2.forward(torch.from_numpy(X.astype(np.float32))))) - np.round(y.astype(np.float32)))))


# torch.save(net.state_dict(), "./net.pth")

plt.plot(res_loss, label="Relu model")
plt.plot(res_loss2, label="Sigmoid model")
plt.legend()
plt.show()
