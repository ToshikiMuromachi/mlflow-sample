import torch
import torch.nn.functional as F


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # Negative Log Likelihood(NLL) Loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
            if args.dry_run:
                break
        # train_num += labels.size(0)
        # train_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    return train_loss


def valid(model, device, test_loader):
    model.eval()
    correct = 0
    running_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss = running_loss / len(test_loader)

    print('\nvalid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    return valid_loss


def learning(args, epochs, model, train_loader, test_loader, optimizer, scheduler, device):
    result = dict()
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(1, epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        valid_loss = valid(model, device, test_loader)
        scheduler.step()

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    # model save
    if args.save_model:
        torch.save(model.state_dict(), "../../model/mnist_cnn.pt")

    # log
    # print(f"Epoch {ep + 1} loss = {train_loss:.06} val_acc = {val_acc:.04}")
    return train_loss_list, valid_loss_list
