# Feature Selection for Digit Classification using Greedy and Mutual Information Methods

## Research Overview
This research project focuses on identifying the **most influential features** in the **`digits` dataset** using two distinct feature selection algorithms: a **Greedy Algorithm** and the **Mutual Information (MI)** method. The dataset, sourced from the `sklearn` library, contains images of handwritten digits (0-9) represented by 64 features, with each feature corresponding to the pixel intensity values of an 8x8 image.

For more details on the dataset, refer to the official [Scikit-learn Digits Dataset Documentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html).

### Objective
The objective of this research is to systematically evaluate and identify a subset of the **top 5 most important features** from the `digits` dataset using two different approaches and compare their performance in classifying the digits.

## Methodology
### Dataset Description
- The `digits` dataset consists of 64 features for each image, with each feature having a value between 0 and 16.
- The target variable, `y`, has 10 distinct classes (0-9), corresponding to the digits.

### Feature Selection Techniques
1. **Greedy Algorithm**  
   - Start with an empty set of features.
   - Iteratively add the feature that results in the lowest classification error when combined with the current set of selected features.
   - Stop when the subset reaches 5 features.

2. **Mutual Information (MI)**  
   - For MI calculation, each feature value is grouped into three bins:  
     - **0-4**: Representing low intensity (White)
     - **5-10**: Representing medium intensity (Gray)
     - **11-16**: Representing high intensity (Black)
   - Use `sklearn`'s `mutual_info_classif` to rank the features based on the mutual information score.
   - Select the top 5 features based on the highest MI values.

### Classification Model
The selected features are used in a **Logistic Regression** model configured for **multiclass classification**. Key parameters to optimize:
- **Penalty**: Set to `l2` to avoid overfitting and enforce regularization.
- **Regularization Strength (`C`)**: Determine the optimal `C` value by testing multiple configurations to minimize classification error.

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy on the test set.
- **Feature Importance**: Contribution of each selected feature to the classification decision.

## Implementation
The project uses Python and the following libraries:
- `sklearn` for feature selection, classification, and evaluation.
- `numpy` for data manipulation.
- `matplotlib` for visualizations.
