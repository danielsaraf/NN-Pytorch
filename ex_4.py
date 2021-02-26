import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as functional
from torchvision import transforms
from torchvision import datasets

EPOCHS = 10
input_size = 784
BATCH_SIZE = 64
LR = 0.01
MODEL_D_LR = 0.1

# Models
class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.name = "model A"
        self.image_size = input_size
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = optimizer.SGD(self.parameters(), lr=LR)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = functional.relu(self.fc0(x))
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.name = "model B"
        self.image_size = input_size
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = optimizer.Adam(self.parameters(), lr=LR)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = functional.relu(self.fc0(x))
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


class ModelC(nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.name = "model C"
        self.image_size = input_size
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = optimizer.SGD(self.parameters(), lr=LR)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = functional.relu(self.fc0(x))
        x = functional.dropout(x, training=self.training)
        x = functional.relu(self.fc1(x))
        x = functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.name = "model D"
        self.image_size = input_size
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = optimizer.SGD(self.parameters(), lr=MODEL_D_LR)
        self.batch_normalize_100 = nn.BatchNorm1d(100)
        self.batch_normalize_50 = nn.BatchNorm1d(50)
        self.batch_normalize_10 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.batch_normalize_100(self.fc0(x))
        x = functional.relu(x)
        x = self.batch_normalize_50(self.fc1(x))
        x = functional.relu(x)
        x = self.batch_normalize_10(self.fc2(x))
        return functional.log_softmax(x, dim=1)


class ModelE(nn.Module):
    def __init__(self):
        super(ModelE, self).__init__()
        self.name = "model E"
        self.image_size = input_size
        self.fc0 = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.optimizer = optimizer.Adam(self.parameters(), lr=LR)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = functional.relu(self.fc0(x))
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = functional.relu(self.fc4(x))
        x = self.fc5(x)
        return functional.log_softmax(x, dim=1)


class ModelF(nn.Module):
    def __init__(self):
        super(ModelF, self).__init__()
        self.name = "model F"
        self.image_size = input_size
        self.fc0 = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.optimizer = optimizer.Adam(self.parameters(), lr=LR)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return functional.log_softmax(x, dim=1)


def get_data():
    # I found that the mean = 0.1307 and the standard deviation = 0.3081
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    # download and normalize (zscore) the training_set
    training_set = datasets.FashionMNIST('/.dataset', train=True, download=True,transform=transform)
    # divide training_set to training set and validation set [80:20]
    training_len = int(len(training_set) * 0.8)
    validation_len = int(len(training_set) * 0.2)
    training_set, validation_set = torch.utils.data.random_split(training_set, [training_len, validation_len])
    # create training_loader (shuffle and set batch size to BATCH_SIZE)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=True)
    # load test set and create test_loader
    test_set = datasets.FashionMNIST('/.dataset', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set)
    # load test_x from file and zscore normalize it
    test_x = np.loadtxt(sys.argv[3])
    test_x = ((test_x/255) - 0.1307) / 0.3081
    test_x = torch.tensor(test_x).float()

    return training_loader, validation_loader, test_loader, test_x


def train_model(model, training_loader):
    model.train()
    train_loss = 0
    correct = 0
    # iterate once over training_loader (1 epoc)
    for batch_idx, (data, labels) in enumerate(training_loader):
        model.optimizer.zero_grad()
        output = model(data)
        loss = functional.nll_loss(output, labels)
        loss.backward()
        model.optimizer.step()
        # calculate loss and accuracy for report file
        train_loss += loss
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).cpu().sum()
    train_loss /= len(training_loader.dataset) / BATCH_SIZE
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(training_loader.dataset),
        100. * correct / len(training_loader.dataset)))


def test_model(model, validation_loader):
    # iterate over validation_loader and calculate loss and accuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)
            test_loss += functional.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(validation_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))


def test_models(models_to_test, training_loader, validation_loader):
    # test loss and accuracy in each model in models_to_test
    for model in models_to_test:
        print("==================== model: " + model.name + " ===========================")
        for epoc in range(EPOCHS):
            print("epoc number: " + str(epoc + 1))
            train_model(model, training_loader)
            test_model(model, validation_loader)


def write_predicts(model, test_x):
    # get predication for each picture in test_x and write its prediction to test_y
    predictions_arr = []
    model.eval()
    with torch.no_grad():
        for data in test_x:
            y_hat = model(data)
            predictions_arr.append(y_hat.max(1, keepdim=True)[1].item())

    with open('test_y', 'w') as out:
        for idx, p in enumerate(predictions_arr):
            if (idx < len(predictions_arr) - 1):
                out.write(str(p) + "\n")
            else:
                out.write(str(p))


def main():
    training_loader, validation_loader, test_loader, test_x = get_data()
    # this part is for the report
    # models_to_test = []
    # models_to_test.append(ModelA())
    # models_to_test.append(ModelB())
    # models_to_test.append(ModelC())
    # models_to_test.append(ModelD())
    # models_to_test.append(ModelE())
    # models_to_test.append(ModelF())
    # test_models(models_to_test, training_loader, validation_loader)
    # test_models(models_to_test, training_loader, test_loader)

    # this part is for create test_y predications file
    best_model = ModelD()
    for epoc in range(EPOCHS):
        train_model(best_model, training_loader)
    write_predicts(best_model, test_x)


if __name__ == '__main__':
    main()
