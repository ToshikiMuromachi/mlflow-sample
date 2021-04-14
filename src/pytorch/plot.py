import matplotlib.pyplot as plt
from numpy.random import randint


def plot_learning(train_loss_list, valid_loss_list, save_path):
    # plot learning curve
    num_epochs = len(train_loss_list)
    print(num_epochs)
    fig = plt.figure()
    plt.plot(range(num_epochs), train_loss_list, 'r-', label='train_loss')
    plt.plot(range(num_epochs), valid_loss_list, 'b-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()

    fig.savefig(save_path)
