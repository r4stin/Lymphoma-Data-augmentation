"""
    This script contains the EarlyStopping class which
    is used to stop training when the validation loss
    stops decreasing and save the best model weights.

"""
import copy
import torch


class EarlyStopping:
    def __init__(self, patience=10, best_model_weights=None, best_loss=float('inf')):
        self.patience = patience
        self.best_model_weights = best_model_weights
        self.best_loss = best_loss

    """
        This method is used to stop the training when the
        validation loss stops decreasing for a certain number
        of epochs.
        
    """
    def early_stopping(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.patience = 10  # Reset patience counter
            return True

        else:
            self.patience -= 1
            print("Patience:", self.patience)
            if self.patience == 0:
                print("Stopped!")
                return False

            return True

    """
        This method is used to save the best model weights
    """
    def save_best_model(self, save_path):
        if self.best_model_weights is not None:
            torch.save(self.best_model_weights, save_path)
            print(f"Best model weights saved to {save_path}")
        else:
            print("No best model weights to save.")
