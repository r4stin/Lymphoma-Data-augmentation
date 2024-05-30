#  This is the main script that will be used to train the models with different augmentation techniques.
from Utils.script import *
from Utils.Visualization import *
from Utils.train_test import *
import argparse
import importlib




class Models:

    def Alexnet_model():
        alexnet_model = torchvision.models.alexnet(weights='DEFAULT')
        alexnet_model.classifier[4] = nn.Linear(4096, 1024)
        alexnet_model.classifier[6] = nn.Linear(1024, 3)
        return alexnet_model

    def Resnet_50_model():
        resnet50_model = torchvision.models.resnet50(weights='DEFAULT')
        resnet50_model.fc = nn.Linear(2048, 3)
        return resnet50_model

    def Densenet_121_model():
        densenet121_model = torchvision.models.densenet121(weights='DEFAULT')
        densenet121_model.classifier = nn.Linear(1024, 3)
        return densenet121_model

    models = [Alexnet_model, Resnet_50_model, Densenet_121_model]


if __name__ == '__main__':

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Define command-line arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Size of each batch')
    parser.add_argument('--transform', type=str, default=[], help='Transformation approach')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--augment', type=str, default="False", help='Augmentation flag')

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    transform = args.transform
    augment = args.augment
    learning_rate = args.learning_rate

    if augment == "True":
        transform_list = importlib.import_module(f'Approaches.{transform}').__getattr__()
        flag = True
    elif augment == "False":
        transform_list = []
        transform = "APP0"
        flag = False
    else:
        raise ValueError("Augment flag should be either True or False")

    dm = MyDataModule(batch_size=BATCH_SIZE, transform=transform_list, augment=flag)

    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()
    """
        Uncomment the following line to visualize the  augmented images
    """
    # app_sample_visualizer(train_loader)

    for model in Models.models:
        print(f'Training {model.__name__} model with {transform} transformation')
        model_path = f'./Weights/{model.__name__}_{transform}.pth'
        model = model()
        tt = Train_Test(NUM_EPOCHS, model, train_loader, val_loader, test_loader, model_path)
        tt.setup(learning_rate)
        train_acc, val_acc, train_loss, val_loss = tt.train_val()
        tt.test()

        """
            Uncomment the following line to visualize the results
            (Loss and Accuracy over epochs)
            
        """
        # results_visualizer(train_loss, train_acc, val_loss, val_acc)


