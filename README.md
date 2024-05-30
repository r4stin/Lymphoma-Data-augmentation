# Lymphoma-Data-augmentation
A classic problem in the classification of medical images is the small number of examples for setting the parameters of the classification method. The goal is to implement an algorithm to create "artificial" images starting from the original ones.



## Instructions

1.Create and navigate to the Dataset directory:

	$ mkdir Dataset && cd Dataset

2. Move your dataset file (dataset.mat) into the Dataset directory.

3. Navigate back to the root directory:

	$ cd ..

4. Apply your desired approach between APP1 - APP8.


** Example Usage **

To run the script with specific parameters, use one of the following commands:

With augmentation:	

	$ python main.py --batch_size 32 --transform APP1 --epochs 100 --learning_rate 0.0001 --augment True

Without augmentation:

	$ python main.py --batch_size 32 --epochs 100 --learning_rate 0.0001 --augment False

** Alternative Way Using Jupyter Notebook **

1. Follow steps 1 - 3 from instrusction.

2. In main.ipynd replace "APP1" with your desired approach.



Note: The default values are:

	batch size: 32
	epochs: 100
	learning rate: 0.0001

- You can modify the batch size, number of epochs, and learning rate as needed.

- To install the necessary Python libraries for this project, you can run the following command in your terminal:

        $ pip install argparse importlib sklearn scipy pytorch-lightning torchvision torch numpy matplotlib

- You can access the BEST weights of all models [here](https://bit.ly/Lymphoma-project-weights)
