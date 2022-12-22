Artificial Neural Network Classifier in TensorFlow/Keras for Multi-class Classification - Base problem category as per Ready Tensor specifications.

- artificial neural network
- ANN
- multi-class classification
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- binary classification
- tensorflow
- keras

This is a Multi-class Classifier that uses Artificial Neural Network Classifier implemented through TensorFlow and Keras. It has 2 hidden layers.

We use stochastic gradient descent (SGD) to optimize the loss of this model. We apply softmax to the last layer to output a probability distribution on the K target classes.

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

Hyperparameter Tuning (HPT) is conducted by finding the optimal l1 and l2 regularization values as well as the optimal learning rate for SGD.

During the model development process, the algorithm was trained and evaluated on a variety of publicly available datasets such as email primary-tumor, splice, stalog, steel plate fault, wine, and car.

This Multi-class Classifier is written using Python as its programming language. TensorFlow and Keras is used to implement the main algorithm. Scikitlearn is used in the data preprocessing pipeline and model evaluation. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.
