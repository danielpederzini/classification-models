# Classification From Scratch
A hands-on machine learning project building classification models from scratch and comparing them on multiple datasets with visual analyses.

## Project Overview
This repository explores how different classifiers learn decision boundaries, handle feature relationships, and perform under different data conditions. The goal is to build intuition, not only report metrics.

The project includes:
* Implementation of multiple models from scratch, using only Pandas / Numpy
* Comparison across three datasets with different characteristics
* Confusion matrices, ROC curves, and Precision-Recall curves
* Decision boundary visualization on the Two Moons dataset
* Custom visualizations to understand model behavior

## Datasets
### 1. Breast Cancer Dataset
A structured binary classification dataset used to compare model performance on real tabular medical data.

**Purpose:**
* Evaluate classical classifiers on a common benchmark

![Breast Cancer Dataset](images/cancer.png)

### 2. Titanic Dataset
A real-world dataset with mixed feature types and missing values.

**Purpose:**
* Test how models handle messy, practical data

![Titanic Dataset](images/titanic.png)

### 3. Two Moons Dataset
A synthetic nonlinear dataset used to visualize decision boundaries.

**Purpose:**
* Show which models can learn nonlinear separations

![Two Moons Dataset](images/two_moons.png)

## Models Implemented
The following models were implemented and compared:
* Naive Bayes (NB)
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Logistic Regression
* Artificial Neural Network (ANN)
* Support Vector Machine (SVM)

## What Each Model Contributes
* **Naive Bayes:** Fast probabilistic baseline with strong independence assumptions
* **KNN:** Instance-based learner that predicts using nearby samples
* **Decision Tree:** Rule-based model that splits the feature space recursively
* **Random Forest:** Ensemble of trees that reduces overfitting through averaging
* **Logistic Regression:** Linear classifier with probabilistic output
* **ANN:** Flexible nonlinear model trained through backpropagation
* **SVM:** Margin-based classifier that can model nonlinear boundaries with kernels

## Experiments and Visualizations

### 1. Confusion Matrices
Confusion matrices were generated for every model and dataset to inspect class-wise mistakes.

![Confusion Matrix Example](images/conf_matrix_example.png)

### 2. ROC Curves
ROC curves were used to compare threshold-based performance.

![ROC Curve Example](images/roc_curve_example.png)

### 3. Precision-Recall Curves
Precision-Recall curves were included to better evaluate performance on class imbalance.

![Precision-Recall Curve Example](images/pr_curve_example.png)

### 4. Decision Boundaries on Two Moons
Decision boundaries were plotted for all models on the Two Moons dataset to show how each classifier separates nonlinear data.

![Decision Boundary Example](images/moons_boundary_dt.png)
![Decision Boundary Example](images/moons_boundary_lr.png)
![Decision Boundary Example](images/moons_boundary_svm.png)

### 5. KNN Neighborhood Visualization
For KNN on the Two Moons dataset, the neighborhood of a random test point was visualized to show which training samples influence the prediction.

![Neighborhood Visualization Example](images/knn_neighbor_viz.png)

### 6. Tree Visualizations
Decision Tree and Random Forest structures were visualized to show how split-based models make predictions.

![Tree Visualization Example](images/random_forest_trees.png)

### 7. Loss Curves
For Logistic Regression and ANN, loss over epochs was plotted to show the learning process.

![Loss Curve Example](images/loss_curve_ann_moons.png)

### 8. Feature Importance
For logistic regression, feature coefficients were visualized to show 
which features contribute most to predictions.

![Feature Importance Example](images/feature_importance_lr_cancer.png)

### 9. ANN Activations
For the ANN, activations from all layers were visualized for a random test sample to show how the network processes information.

![ANN Activations Example](images/ann_activation_titanic.png)

### 10. SVM Support Vectors
For SVM on the Two Moons dataset, support vectors were highlighted in the decision boundary plot to show which samples are critical.

![Support Vectors Example](images/svm_support_vectors_viz.png)

## Results Summary
This project highlights how different algorithms behave under different assumptions:

* Linear models work best when the boundary is simple
* Tree-based models handle nonlinear interactions well
* KNN can model complex shapes but may be sensitive to local noise
* Naive Bayes is simple and fast, but its assumptions can limit performance
* ANN and SVM can model more complex patterns, depending on tuning and preprocessing

#### Cancer Dataset Results
| Model               | F1 Score      | ROC AUC      | PR AUC         |
| ------------------- | ------------: | -----------: | -------------: |
| Naive Bayes         |        0.8989 |       0.9794 |         0.9758 |
| KNN                 |        0.9070 |       0.9149 |         0.9500 |
| Decision Tree       |        0.9451 |       0.9500 |         0.9623 |
| Random Forest       |        0.9565 |       0.9809 |         0.9816 |
| Logistic Regression |        0.9348 |       0.9816 |         0.9745 |
| ANN                 |        0.9670 |       0.9924 |         0.9905 |
| SVM                 |        0.9556 |       0.9574 |         0.9750 |

#### Two Moons Dataset Results
| Model               | F1 Score      | ROC AUC      | PR AUC         |
| ------------------- | ------------: | -----------: | -------------: |
| Naive Bayes         |        0.8564 |       0.9456 |         0.9440 |
| KNN                 |        0.9657 |       0.9653 |         0.9770 |
| Decision Tree       |        0.9173 |       0.9677 |         0.9697 |
| Random Forest       |        0.9637 |       0.9900 |         0.9905 |
| Logistic Regression |        0.8550 |       0.9457 |         0.9431 |
| ANN                 |        0.9633 |       0.9895 |         0.9862 |
| SVM                 |        0.9685 |       0.9674 |         0.9760 |

#### Titanic Dataset Results
| Model               | F1 Score      | ROC AUC      | PR AUC         |
| ------------------- | ------------: | -----------: | -------------: |
| Naive Bayes         |        0.7328 |       0.8601 |         0.7906 |
| KNN                 |        0.6885 |       0.7573 |         0.7557 |
| Decision Tree       |        0.7717 |       0.8897 |         0.8549 |
| Random Forest       |        0.7642 |       0.8910 |         0.8594 |
| Logistic Regression |        0.7368 |       0.8684 |         0.8010 |
| ANN                 |        0.7333 |       0.8657 |         0.8159 |
| SVM                 |        0.7520 |       0.8045 |         0.8035 |

### Usage
1. Clone the repository
2. Run the notebooks
3. Play with the hyperparameters and visualizations

**Author**: Daniel Pederzini  
**Purpose**: Machine Learning Educational Project
