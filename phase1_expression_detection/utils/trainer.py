import torch
from utils.metrics import accuracy

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_acc = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            outputs = self.model(images)

            # Loss
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        return avg_loss, avg_acc

    def validate(self, dataloader): 
        self.model.eval()
        total_loss = 0
        total_acc = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                total_acc += accuracy(outputs, labels)

                if batch_idx % 50 == 0:
                    print(f"Batch [{batch_idx}/{len(dataloader)}]")


        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        return avg_loss, avg_acc