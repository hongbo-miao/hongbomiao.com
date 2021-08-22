from evaluate import evaluate
from model.data_loader import train_data_loader, val_data_loader
from model.net import net
from torch import nn, optim
from utils.device import device
from args import get_args
import wandb


def main():
    args = get_args()

    with wandb.init(
        entity="hongbo-miao", project="convolutional-neural-network", config=args
    ) as wb:
        config = wb.config

        net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)
        # optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

        for epoch in range(config.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_data_loader, 0):
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
                    print(
                        "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0
            val_acc = evaluate(net, val_data_loader)
            wb.log(
                {
                    "epoch": epoch,
                    "val_acc": val_acc,
                }
            )

        print("Finished Training")


if __name__ == "__main__":
    main()
