# PixelPatrol

PixelPatrol is a project focused on training a Discriminator model, a critical component of a Generative Adversarial Network (GAN), using the CIFAR-10 dataset. The primary objective of this model is to distinguish between real images from the CIFAR-10 dataset and fake images generated using random noise.

## Project Overview

Generative Adversarial Networks (GANs) are composed of two adversarial models: a Generator and a Discriminator. The Generator creates images, while the Discriminator evaluates these images and classifies them as either "real" (sourced from the actual dataset) or "fake" (produced by the Generator). PixelPatrol focuses on the Discriminator's role, training it to accurately identify real and fake images.

### Key Components

#### Data Loading
- **Dataset**: CIFAR-10, containing 60,000 32x32 color images across 10 classes, with 6,000 images per class.
- **Preprocessing**: Images are scaled to the range [-1, 1] to facilitate model training.

#### Discriminator Model
- **Architecture**: Built using Keras' Sequential API.
- **Layers**: Composed of convolutional layers followed by LeakyReLU activation functions.
- **Output**: A single neuron with a sigmoid activation function, which outputs the probability that an input image is real.

#### Sample Generation
- **Real Samples**: Randomly selected from the CIFAR-10 dataset, labeled as 1 (real).
- **Fake Samples**: Generated using random noise, labeled as 0 (fake).

#### Training
- **Process**: The Discriminator is iteratively trained on a mixture of real and fake samples.
- **Performance Monitoring**: The model's accuracy in distinguishing real from fake images is tracked throughout the training process.

## Installation and Requirements

To set up and run this project, ensure that the following dependencies are installed in your Python environment:

```bash
pip install keras tensorflow matplotlib numpy
```

## How to Run the Project

1. Clone the Repository:

```bash
git clone [https://github.com/your-username/PixelPatrol.git](https://github.com/Mihawk1891/PixelPatrol.git)
cd PixelPatrol
```

2. Run the Jupyter Notebook:

Open the `PixelPatrol.ipynb` notebook in Jupyter and execute all cells to initiate the training of the Discriminator model.

## Project Logic

### Discriminator Model

The Discriminator model processes input images through multiple convolutional layers to extract relevant features. These features are then flattened and passed through dense layers to produce an output that predicts whether the input is real or fake.

### Training Loop

The Discriminator is trained in a loop, processing batches of real and fake images. The model's weights are adjusted during each iteration to improve its ability to differentiate between real and fake images.

### Accuracy Tracking

During training, the model's accuracy in correctly classifying images is continuously monitored. This helps in evaluating the model's performance and effectiveness.

## Results and Observations

The final trained Discriminator model demonstrates a strong ability to distinguish between real and fake images. However, the accuracy of the model can vary based on the number of training iterations and the specific architecture of the Discriminator.

## Future Work

- **Integration with Generator**: To complete the GAN framework, a Generator model should be developed and trained alongside the Discriminator.
- **Hyperparameter Tuning**: Experiment with different architectures and hyperparameters to enhance the Discriminator's performance and accuracy.
