# Potato Disease Detection

This project focuses on detecting potato diseases using a deep learning model. The model is trained to classify potato leaves into three categories: Early blight, Late blight, and Healthy.

## Table of Contents

- Dataset
- Dependencies
- Model Architecture
- Training
- Evaluation
- Usage
- Results

## Dataset

The dataset used for this project is available on Kaggle: [Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease). It contains images of potato leaves categorized into three classes:

- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

The dataset is split into training, testing, and validation sets.

## Dependencies

The following libraries are required to run the project:

- TensorFlow
- NumPy
- Seaborn
- Matplotlib

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture includes:

- Rescaling and resizing layers
- Data augmentation layers (random flip and rotation)
- Multiple convolutional layers with ReLU activation
- Max pooling layers
- Fully connected layers with ReLU and Softmax activation

## Training

The model is trained for 50 epochs with a batch size of 32. The optimizer used is Adam, and the loss function is sparse categorical crossentropy. The training process includes validation on a separate validation set.

## Evaluation

The model is evaluated on a test set, and the accuracy is calculated. The model achieves high accuracy in classifying the potato leaves into the correct categories.

## Model Loading
You can load the model using this code.

```python
model = tf.keras.models.load_model('Model')
```

## Usage

To use the model for prediction, you can load the saved model and use the `predictor` function. The function takes an image as input and returns the predicted class and confidence score.

```python
def predictor(model, image):
    classes = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predict_arr = model.predict(img_array, verbose=0)
    prediction = np.argmax(predict_arr[0])
    confidence = round(predict_arr[0][prediction] * 100, 2)
    label = classes[prediction]
    return label, confidence
```
# Results
The model achieves high accuracy in classifying potato leaves into the correct categories. The training and validation accuracy plots show that the model performs well on both the training and validation sets. The model is also evaluated on a test set, and the accuracy is calculated.

- Training and Validation Accuracy : 98%
- Training and Validation Loss : 0.018
