#based on the pytorch example https://github.com/pytorch/examples/tree/main/mnist
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from cnn import Net

class Predictor():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        try:
            self.model.load_state_dict(torch.load("mnist_cnn.pt"))
        except:
            print("Weights file not found")
        
        
    def predict(self, x) -> torch.tensor:
        """
        Parameters
        ----------
        x : input image tensor 1 x 1 x 28 x 28

        Returns model output in softmax form 
        """
        self.model.eval()
        with torch.no_grad():
            prep = transforms.Normalize((0.1307,), (0.3081,))
            x = prep(x)
            x = x.to(self.device)
            return torch.exp(self.model(x))
        

    def train(self, args, train_loader, optimizer, epoch):
        #train model
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    
    
    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(test_loader.dataset)
    
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
    
                
    
    def retrain_weights(self):
        # Training settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=20, metavar='N',
                            help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-self.model', action='store_true', default=True,
                            help='For Saving the current self.model')
        args = parser.parse_args()
        
        print(args.epochs)
        
        torch.manual_seed(args.seed)
    
        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
    
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        dataset_training = datasets.MNIST('../data', train=True, download=True,
                           transform=transform)
        dataset_test = datasets.MNIST('../data', train=False,
                           transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset_training, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    
        self.model = Net().to(self.device)
        
        optimizer = optim.Adadelta(self.model.parameters(), lr=args.lr)
    
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        
        for epoch in range(1, args.epochs + 1):
            self.train(args, train_loader, optimizer, epoch)
            test(test_loader)
            scheduler.step()
            if args.save_self.model:
                torch.save(self.model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    predictor1 = Predictor()
    predictor1.retrain_weights()