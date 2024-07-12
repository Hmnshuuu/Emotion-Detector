# Emotion Detector

## Description

This project focuses on detecting emotions using Deep Learning, specifically Convolutional Neural Networks (CNN). The goal is to create an efficient and accurate system that can recognize different emotions from facial expressions in images.

## Key Skills

- Deep Learning
- Convolutional Neural Networks (CNN)
- Kaggle
- Python

## Project Link

[Emotion Detector](https://github.com/Hmnshuuu/Emotion-Detector)

## Project Details

I developed this Deep Learning Model using CNN. The project involves training four different models:

1. **Simple CNN**
2. **CNN with Augmentation**
3. **VGG16 with Transfer Learning**
4. **ResNet50 with Transfer Learning**

In this model, I also used techniques like Batch Normalization, Dropout, and Pooling to improve the results. After training all four models, the ResNet50 model achieved the best accuracy.

## Features

- **Multiple Model Architectures**: Comparison of different CNN architectures including VGG16 and ResNet50 with transfer learning.
- **Advanced Techniques**: Utilizes Batch Normalization, Dropout, and Pooling for better performance.
- **High Accuracy**: The ResNet50 model demonstrated the highest accuracy among all tested models.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Hmnshuuu/Emotion-Detector.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Emotion-Detector
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Train the models:
    ```bash
    python train.py --model simple_cnn
    python train.py --model cnn_with_augmentation
    python train.py --model vgg16_transfer_learning
    python train.py --model resnet50_transfer_learning
    ```

2. Evaluate the models:
    ```bash
    python evaluate.py --model resnet50_transfer_learning
    ```

3. Run the emotion detector:
    ```bash
    python detect_emotion.py --image path/to/your/image.jpg
    ```

## Project Structure

- `train.py`: Script for training the different CNN models.
- `evaluate.py`: Script for evaluating the trained models.
- `detect_emotion.py`: Script for running the emotion detection on images.
- `models/`: Directory containing the model definitions.
- `data/`: Directory for storing the datasets.

## Dataset

Prepare your dataset and place it in the `data/` directory. The dataset should contain labeled images of facial expressions representing different emotions.

## Training the Models

To train the models on your dataset, follow these steps:

1. Prepare your dataset and place it in the `data/` directory.
2. Run the training scripts for each model as shown in the Usage section.
3. The trained models will be saved in the specified directory.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for the deep learning framework.
- The open-source community for their invaluable resources and contributions.

---

Feel free to reach out if you have any questions or need further assistance. Happy coding!

[Himanshu]  
[himanshujangid364@gmail.com]  
[LinkedIn](https://www.linkedin.com/in/himanshuuu/)
