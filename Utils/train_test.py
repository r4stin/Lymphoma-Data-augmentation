"""
    This file contains the Train_Test class which is used to train and test the model.
    The class contains the following methods:
    1. setup: This method is used to initialize the device (GPU or CPU), criterion, and optimizer.
    2. train_val: This method is used to train the model and validate it.
    3. test: This method is used to test the model on the test data.

"""
import torch
from Utils.EarlyStopping import *
from torch import nn, optim
from sklearn.metrics import classification_report
import itertools


class Train_Test:

    def __init__(self, num_epoch, model, train_loader, val_loader, test_loader, model_path):
        self.model = model
        self.NUM_EPOCHS = num_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_path = model_path

    def setup(self, learning_rate):

        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model = self.model.to(self.DEVICE)

    def train_val(self):

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        es = EarlyStopping(patience=10, best_model_weights=None, best_loss=float('inf'))

        # Training loop
        for epoch in range(self.NUM_EPOCHS):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_accuracy = correct_predictions / total_samples

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            # Validation phase
            self.model.eval()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            val_loss = running_loss / len(self.val_loader.dataset)
            val_losses.append(val_loss)
            val_accuracy = correct_predictions / total_samples
            val_accuracies.append(val_accuracy)

            print(f"Epoch [{epoch + 1}/{self.NUM_EPOCHS}] - Train loss: {epoch_loss},--- Validation loss: {val_loss}")
            print(f'Train Accuracy: {100 * epoch_accuracy:.2f}%,--- Val Accuracy: {100 * val_accuracy:.2f}%')

            if epoch > 20:
                if not es.early_stopping(val_loss, self.model):
                    break

            print('--------------------------')

            print()
            print()

        es.save_best_model(self.model_path)

        print('Training Finished')
        return train_losses, train_accuracies, val_losses, val_accuracies

    def test(self):
        correct = 0
        total = 0
        predicted_list = []
        true_val = []

        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path))

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicted_list.append(predicted.cpu().numpy())
                true_val.append(labels.cpu().numpy())

        predicted_list = list(itertools.chain(*predicted_list))
        true_val = list(itertools.chain(*true_val))

        print(classification_report(true_val, predicted_list))
        print(f'Accuracy of the network on the test images: {100 * correct // total} %')

