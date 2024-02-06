# Artificial-Intelligence-Course-F2024
* CA1. [Genetic Algorithm for Curve Fitting](https://github.com/MobinaMhr/Artificial-Intelligence-Course-F2024/tree/main/CA1-Genetic)
* CA2. [Reinforcement Learning](https://github.com/MobinaMhr/Artificial-Intelligence-Course-F2024/tree/main/CA2-Reinforcement-Learning)
* CA3. [Hidden Markov Model](https://github.com/MobinaMhr/Artificial-Intelligence-Course-F2024/tree/main/CA3-HMM)
* CA4. [Machine Learning](https://github.com/MobinaMhr/Artificial-Intelligence-Course-F2024/tree/main/CA4-ML)
* CA5. [Convolutional Neural Networks](https://github.com/MobinaMhr/Artificial-Intelligence-Course-F2024/tree/main/CA5-CNN)
* CA6. [Natural Language Processing](https://github.com/MobinaMhr/Artificial-Intelligence-Course-F2024/tree/main/CA6-NLP)


## CA1. Genetic Algorithm for Curve Fitting

This project implements a genetic algorithm to solve the curve-fitting problem. The goal is to determine the coefficients of a polynomial equation based on a given set of data points. The genetic algorithm utilizes concepts inspired by nature and natural selection, aiming to evolve a population of potential solutions over generations to find the optimal solution.


## CA2. Reinforcement Learning

In this project, the primary focus is on implementing reinforcement learning algorithms, specifically, the Iteration Value and Iteration Policy, which are analyzed and implemented for the FrozenLake-v1 environment. Additionally, a Q-learning agent is implemented for the Taxi-v3 environment. The project also delves into investigating the impact of decreasing learning rates in the Q-learning algorithm and provides a demonstration of its effects.

## CA3. Hidden Markov Model

This project aims to classify music into four distinct genres—blues, pop, metal, and hip-hop—by implementing a first-order Hidden Markov Model. 
The audio MFCCs were extracted and then a model was implemented using the Expectation-Maximization (EM) algorithm.

## CA4. Machine Learning

In this project, various machine learning models were implemented and evaluated across multiple phases. The initial steps involved dataset preprocessing and splitting into training and testing sets. Linear Regression was implemented, followed by Multiple Regression models with different feature combinations. Classification tasks utilized Decision Tree, K-Nearest Neighbors, and Logistic Regression, with hyperparameter tuning using GridSearchCV for optimization.

A comparative analysis between Decision Tree and Random Forest models was conducted to examine bias and variance. The project also explored the impact of noise on the dataset, specifically on the Decision Tree model's accuracy. Gradient Boosting, a sequential improvement technique, and XGBoost classifier were introduced, emphasizing their accuracy and the identification of optimal parameters through experimentation.


## CA5. Convolutional Neural Networks

In this project, a Convolutional Neural Network (CNN) is implemented using PyTorch for the classification of MRI brain images into four categories: no-tumor, meningioma, glioma, and pituitary. The project involves the creation of a specialized dataset class (BrainTumorDataset), the definition of a neural network model with convolutional and fully connected layers, and the use of optimization techniques such as Adam and SGD, along with regularization methods including Batch Normalization and Dropout. The primary objective is the accurate classification of brain tumor images.


## CA6. Natural Language Processing

In the first phase, data preprocessing was performed on news articles using the Hazm library, including normalization, punctuation removal, tokenization, stopword removal, lemmatization, and handling empty tokens. In the second phase, the dataset was split into train and test sets, tagged with labels, and used to train a Doc2Vec model. Feature vectors were generated, and both KMeans and DBSCAN clustering models were applied and evaluated based on homogeneity and silhouette scores. In the third phase, Principal Component Analysis (PCA) was employed to reduce the dimensionality of the dataset.
