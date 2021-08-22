def train(net, data_loader, optimizer, criterion):
    net.train()
    running_loss = 0.0

    for i, data in enumerate(data_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss
