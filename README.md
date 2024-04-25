# Aim
This assignment aims to use FastAPI to serve a Digit Recognition App.

# Procedure
The procedure for doing this assignment is as follows:-

1) **Training the Model** - Train the 10 models from the MNIST notebook and find the best model based on test set accuracy. It was found that 2nd model is best with an accuracy of 98.10\%.

2) **Loading and Predict using Model** - Make two functions for loading the model and predicting the digit of a given image.

3) **Creating API Endpoint** - Create an API endpoint ``predict/'' to incorporate these functions to make this app available using FastAPI.
