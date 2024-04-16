import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dataset
from model import LeNet5, CustomMLP
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def plot_stats(train_stats, test_stats, title):
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#     epochs = range(len(train_stats['loss']))
#     ax[0].plot(epochs, train_stats['loss'], 'r', label='Train Loss')
#     ax[0].plot(epochs, test_stats['loss'], 'b', label='Test Loss')
#     ax[0].set_title('Loss over Epochs')
#     ax[0].set_xlabel('Epochs')
#     ax[0].set_ylabel('Loss')
#     ax[0].legend()

#     ax[1].plot(epochs, train_stats['accuracy'], 'r', label='Train Accuracy')
#     ax[1].plot(epochs, test_stats['accuracy'], 'b', label='Test Accuracy')
#     ax[1].set_title('Accuracy over Epochs')
#     ax[1].set_xlabel('Epochs')
#     ax[1].set_ylabel('Accuracy')
#     ax[1].legend()

#     plt.suptitle(title)
#     plt.show()

def plot_stats(train_stats, test_stats, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    epochs = range(len(train_stats['loss']))
    ax[0].plot(epochs, train_stats['loss'], 'r', label='Train Loss')
    ax[0].plot(epochs, test_stats['loss'], 'b', label='Test Loss')
    ax[0].set_title('Loss over Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epochs, train_stats['accuracy'], 'r', label='Train Accuracy')
    ax[1].plot(epochs, test_stats['accuracy'], 'b', label='Test Accuracy')
    ax[1].set_title('Accuracy over Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.suptitle(title)
    # 파일로 저장
    plt.savefig(f'{title}.png')
    plt.show()


def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in trn_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    trn_loss = total_loss / len(trn_loader)
    acc = 100. * correct / total
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    tst_loss = test_loss / len(tst_loader)
    acc = 100. * correct / len(tst_loader.dataset)
    return tst_loss, acc

def main():
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 10
    weight_decay=0.001
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train_dir = r'C:\deep\mnist-classification\data'
    test_dir = r'C:\deep\mnist-classification\data'

    train_dataset = dataset.MNIST(data_dir=train_dir, train=True)
    test_dataset = dataset.MNIST(data_dir=test_dir, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet5().to(device)
    num_params = count_parameters(model)
    print(f"LeNet-5 모델의 파라메터 수: {num_params}")

    lenet_train_stats = {'loss': [], 'accuracy': []}
    lenet_test_stats = {'loss': [], 'accuracy': []}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,weight_decay=0.001)


    for epoch in range(num_epochs):
        trn_loss, trn_acc = train(model, train_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(model, test_loader, device, criterion)
        lenet_train_stats['loss'].append(trn_loss)
        lenet_train_stats['accuracy'].append(trn_acc)
        lenet_test_stats['loss'].append(tst_loss)
        lenet_test_stats['accuracy'].append(tst_acc)
        print(f'LeNet-5 Epoch {epoch+1}/{num_epochs}: Train Loss: {trn_loss:.4f} | Train Accuracy: {trn_acc:.2f}% | Test Loss: {tst_loss:.4f} | Test Accuracy: {tst_acc:.2f}%')


    plot_stats(lenet_train_stats, lenet_test_stats, 'LeNet-5 Training and Testing Stats')


    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 10
    #l2 regulazation 을 위한 decay factor
    weight_decay=0.001
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = CustomMLP().to(device)
    num_params = count_parameters(model)
    print(f"CustomMLP 모델의 파라메터 수: {num_params}")
    # loss function 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,weight_decay=weight_decay)

    lenet_train_stats = {'loss': [], 'accuracy': []}
    lenet_test_stats = {'loss': [], 'accuracy': []}


    for epoch in range(num_epochs):
        trn_loss, trn_acc = train(model, train_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(model, test_loader, device, criterion)
        lenet_train_stats['loss'].append(trn_loss)
        lenet_train_stats['accuracy'].append(trn_acc)
        lenet_test_stats['loss'].append(tst_loss)
        lenet_test_stats['accuracy'].append(tst_acc)
        print(f'CustomMLP Epoch {epoch+1}/{num_epochs}: Train Loss: {trn_loss:.4f} | Train Accuracy: {trn_acc:.2f}% | Test Loss: {tst_loss:.4f} | Test Accuracy: {tst_acc:.2f}%')
        
    plot_stats(lenet_train_stats, lenet_test_stats, 'Custom Training and Testing Stats')

if __name__ == '__main__':
    main()