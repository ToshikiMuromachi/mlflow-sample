import torch
import torchvision
import torchvision.transforms as T

def generate_loader(batch_size):
    """
    train_loader, test_loaderを作成
    Parameters
    ----------
    batch_size: int
        バッジサイズ
    Returns
    -------
    train_loader: DataLoader
        train_loader
    test_loader: DataLoader
        test_loader
    """
    
    # trans
    transformer = T.Compose([
        T.ToTensor()
    ])

    # train_loader
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./images', train=True, download=True, 
                                   transform=transformer), 
        batch_size=batch_size, 
        shuffle=True,
    )

    # test_loader
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./images', train=False, transform=transformer), 
        batch_size=batch_size, 
        shuffle=True
    )

    return train_loader, test_loader