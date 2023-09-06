from locale import atof
import numpy as np
import csv
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.functional import normalize
from torcheval.metrics import BinaryAccuracy

size = 500

dtype = np.float32
sttl = np.empty(size, np.float32)
dttl = np.empty(size, np.float32)
spkts = np.empty(size, np.float32)
dpkts = np.empty(size, np.float32)
smeansz = np.empty(size, np.float32)
dmeansz = np.empty(size, np.float32)
stime = np.empty(size, np.float32)
attack = np.empty(size, np.int32)

# Define the class for single layer NN
class one_layer_net(torch.nn.Module):
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer
        # Applies a linear transformation to the incoming data: y = xA^T + b
        # Полностью соединенный слой нейронной сети представлен объектом nn.Linear
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size)
        torch.nn.init.normal_(self.linear_one.weight, 0, 1 / np.sqrt(input_size))
        torch.nn.init.normal_(self.linear_two.weight, 0, 1 / np.sqrt(hidden_neurons))
        # defining layers as attributes
        self.layer_in = None
        self.act = None
        self.layer_out = None
    # prediction function
    # получает входные данные и возвращает результат их трансформации оператором.
    # Он, кроме того, решает внутренние задачи, необходимые для вычисления градиентов.
    def forward(self, x):
        self.layer_in = self.linear_one(x)
        self.act = torch.relu(self.layer_in)
        self.layer_out = self.linear_two(self.act)
        y_pred = torch.relu(self.linear_two(self.act))
        return torch.sigmoid(y_pred)

def train_loop(X, Y, model):
    # Define the training loop
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    epochs = 15
    cost = []
    total = 0
    for epoch in range(epochs):
        epoch = epoch + 1
        for x, y in zip(X, Y):
            x = x.float()
            y = y.float()
            # print(x)
            yhat = model(x)  # give input
            y.resize_as_(yhat)
            # print(yhat)
            # print(y)
            loss = F.binary_cross_entropy(yhat, y)
            # принимает частные производные функции потерь по отношению к выходам оператора и реализует расчёт
            # частных производных функции потерь по отношению к входным данным оператора и к параметрам (если они есть).
            # Для обратного распространения ошибки все, что нам нужно сделать, это вызвать loss.backward().
            # Однако необходимо очистить существующие градиенты, иначе градиенты будут накапливаться в существующих градиентах.
            loss.backward()
            optimizer.step()  # Performs a single optimization step (parameter update).
            optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensors to zero.
            # get total loss
            total += loss.item()
        total = total / size  # len(X)
        print(total)
        cost.append(total)
        if epoch % 5 == 0:
            print(str(epoch) + " " + "epochs done!")  # visualze results after every size epochs

    # plot the cost
    # print(total)
    # plt.plot(cost)
    # plt.xlabel('epochs')
    # plt.title('cross entropy loss')
    # plt.show()

def test_loop(X, Y, model):
    model.eval()
    total = 0
    i = 0
    ypred = np.empty(size, np.float32)
    with torch.no_grad():
        for x, y in zip(X, Y):
            x = x.float()
            y = y.float()
            # print(x)
            yhat = model(x)  # give input
            y.resize_as_(yhat)
            # print(yhat)
            # print(y)
            loss = F.binary_cross_entropy(yhat, y)
            # get total loss
            total += loss.item()
            ypred[i] = yhat
            i += 1
    total = total / size  # len(X)
    metric = BinaryAccuracy()
    ypred = torch.from_numpy(ypred)
    metric.update(ypred, Y)
    metric.compute()
    print("loss = ", str(total))
    print("accuracy = ", str(float(metric.compute())))
    # plt.plot(cost)
    # plt.xlabel('epochs')
    # plt.title('cross entropy loss')
    # plt.show()


def read_csv(name_file):
    print("start reading cvs file...")
    with open(name_file, encoding='utf-8') as r_file:
        # Создаем объект reader, указываем символ-разделитель ","
        file_reader = csv.reader(r_file, delimiter=",")
        # Счетчик для подсчета количества строк и вывода заголовков столбцов
        count = 0
        # Считывание данных из CSV файла
        for row in file_reader:
            i = 0
            sttl[count] = row[0]
            dttl[count] = row[1]
            spkts[count] = row[2]
            dpkts[count] = row[3]
            smeansz[count] = row[4]
            dmeansz[count] = row[5]
            stime[count] = row[6]
            attack[count] = row[7]
            count += 1
    r_file.close()

def main():
    read_csv("train.csv")
    print("training file read")
    X_train = torch.tensor([sttl, dttl, spkts, dpkts, smeansz, dmeansz, stime], dtype=torch.float32)
    X_train.t_()
    X_train = normalize(X_train, p=2.0)
    Y_train = torch.from_numpy(attack)
    # create the model
    print("start training...")
    model = one_layer_net(7, 12, 1)  # 15 represents two neurons in one hidden layer
    train_loop(X_train, Y_train, model)

    read_csv("validate.csv")
    print("validating file read")
    X_val = torch.tensor([sttl, dttl, spkts, dpkts, smeansz, dmeansz, stime], dtype=torch.float32)
    X_val.t_()
    X_val = normalize(X_val, p=2.0)
    Y_val = torch.from_numpy(attack)
    # create the model
    print("start validating...")
    test_loop(X_val, Y_val, model)


if __name__ == '__main__':
    main()
