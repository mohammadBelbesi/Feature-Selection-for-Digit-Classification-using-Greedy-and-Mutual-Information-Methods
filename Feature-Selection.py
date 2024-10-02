import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score
import warnings

optimal_C = 1.5

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

digits = load_digits()
X, y = digits.data, digits.target


splitTo3 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
x_split = splitTo3.fit_transform(X)


def greedy_feature_selection(X, y, num_features):
    important_features = []
    features = list(range(X.shape[1]))
    count = 0
    
    while count < num_features:
        top_score = 0
        top_feature = None
        feature_index = 0
        
        while feature_index < len(features):
            feature = features[feature_index]
            current_features = important_features + [feature]
            X_subset = X[:, current_features]
            
            X_train, X_val, y_train, y_val = train_test_split(X_subset, y, test_size=0.3, random_state=42)
            model = LogisticRegression(penalty='l2', C=optimal_C, solver='liblinear', multi_class='ovr')
            model.fit(X_train, y_train)
            score = accuracy_score(y_val, model.predict(X_val))
            
            if score > top_score:
                top_score = score
                top_feature = feature
            
            feature_index += 1
        
        important_features.append(top_feature)
        features.remove(top_feature)
        count += 1
        
    return important_features


def mutual_Info_feature_selection(X, y, num_features):
    mutual_Info_scores = mutual_info_classif(X, y, discrete_features=True)
    top_features = np.argsort(mutual_Info_scores)[-num_features:]
    return top_features


important_features_greedy = greedy_feature_selection(X, y, 5)


important_features_mi = mutual_Info_feature_selection(x_split, y, 5)


def evaluate_model(X, y, important_features):
    X_subset = X[:, important_features]
    X_train, X_val, y_train, y_val = train_test_split(X_subset, y, test_size=0.3, random_state=42)
    model = LogisticRegression(penalty='l2', C=optimal_C, solver='liblinear', multi_class='ovr')
    model.fit(X_train, y_train)
    score = accuracy_score(y_val, model.predict(X_val))
    return score


greedy_score = evaluate_model(X, y, important_features_greedy)
mutual_Info_score = evaluate_model(X, y, important_features_mi)


print("The important features by using Greedy Algorithm:", important_features_greedy)
print("Accuracy with Greedy Algorithm important features:", greedy_score)

print("The important features by using MI method:", important_features_mi)
print("Accuracy with MI method important features:",  mutual_Info_score)


if greedy_score > mutual_Info_score:
    print("Greedy algorithm performed better .")
else:
    print("MI method performed better .")
