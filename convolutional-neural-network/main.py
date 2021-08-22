from torch import nn, optim

from evaluate import evaluate
from model.data_loader import train_data_loader, val_data_loader
from model.net import net
from utils.device import device


def train(
    net,
    train_data_loader,
    criterion,
    optimizer,
    num_epochs,
):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        evaluate(net, val_data_loader)

    print("Finished Training")


if __name__ == "__main__":
    learning_rate = 0.001
    num_epochs = 2

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    train(
        net,
        train_data_loader,
        criterion,
        optimizer,
        num_epochs,
    )
