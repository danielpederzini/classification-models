import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def split_train_test(x, y, train_size=0.8, random_state=42):
    train_set_size = int(len(x) * train_size)
    
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(x))
        
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:]
    
    return x.iloc[train_indices], x.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

def z_score(x):
    return (x - x.mean()) / x.std(ddof=1)

class CancerDataHelper():
    def load_dataset(one_hot=True, normalize=True):
        cancer_data = pd.read_csv("./input/breast_cancer.csv")
        cancer_data = cancer_data.drop(columns=["id"])
        
        if (one_hot):
            cancer_data = pd.get_dummies(cancer_data, columns=["diagnosis"], drop_first=True, dtype=int)
        
        x_train, x_test, y_train, y_test = split_train_test(cancer_data.drop("diagnosis_M", axis=1), cancer_data["diagnosis_M"])
        
        if (normalize):
            x_train = z_score(x_train)
            x_test = z_score(x_test)
        
        return x_train, x_test, y_train, y_test

    def plot_outcome_distribution(x, y):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        malignant_counts = y.value_counts()
        labels = ["Benign", "Malignant"]
        colors = ["#2ecc71", "#e74c3c"]

        ax.pie(malignant_counts.values, labels=labels, autopct="%1.1f%%", 
            colors=colors, startangle=90, textprops={"fontsize": 12})
        ax.set_title("Class Distribution in Training Data", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

        print(f"Benign: {malignant_counts[0]} ({malignant_counts[0]/len(y)*100:.1f}%)")
        print(f"Malignant: {malignant_counts[1]} ({malignant_counts[1]/len(y)*100:.1f}%)")
        
    def plot_correlation(x, y):
        fig, ax = plt.subplots(figsize=(20, 16))

        correlation_data = x.copy()
        correlation_data["diagnosis_M"] = y
        correlation_matrix = correlation_data.corr()

        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    center=0, square=True, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.show()
        
    def plot_boxplots(x, y):
        fig, axes = plt.subplots(6, 5, figsize=(24, 20))
        axes = axes.ravel()

        viz_data = x.copy()
        viz_data["Outcome"] = y.replace({1: "Malignant", 0: "Benign"})

        for idx, feature in enumerate(x.columns):
            sns.boxplot(data=viz_data, x="Outcome", y=feature, ax=axes[idx], 
                        palette=["#2ecc71", "#e74c3c"])
            axes[idx].set_title(f"{feature} Distribution by Outcome", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Outcome", fontsize=11)
            axes[idx].set_ylabel(feature, fontsize=11)

        plt.tight_layout()
        plt.show()

class TitanicDataHelper():
    def load_dataset(one_hot=True, normalize=True):
        titanic_data = pd.read_csv("./input/titanic_survival.csv")
        titanic_data = titanic_data.drop(columns=["Name", "PassengerId", "Ticket", "Cabin"])

        age_median = titanic_data["Age"].median()
        titanic_data["Age"] = titanic_data["Age"].fillna(age_median)

        embarked_mode = titanic_data["Embarked"].mode()[0]
        titanic_data["Embarked"] = titanic_data["Embarked"].fillna(embarked_mode)

        if (one_hot):
            titanic_data = pd.get_dummies(titanic_data, columns=["Sex", "Embarked"], drop_first=True, dtype=int)
        
        x_train, x_test, y_train, y_test = split_train_test(titanic_data.drop("Survived", axis=1), titanic_data["Survived"])
        
        if (normalize):
            x_train = z_score(x_train)
            x_test = z_score(x_test)
        
        return x_train, x_test, y_train, y_test

    def plot_outcome_distribution(x, y):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        survived_counts = y.value_counts()
        labels = ["Died", "Survived"]
        colors = ["#e74c3c", "#2ecc71"]

        ax.pie(survived_counts.values, labels=labels, autopct="%1.1f%%", 
            colors=colors, startangle=90, textprops={"fontsize": 12})
        ax.set_title("Class Distribution in Training Data", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

        print(f"Died: {survived_counts[0]} ({survived_counts[0]/len(y)*100:.1f}%)")
        print(f"Survived: {survived_counts[1]} ({survived_counts[1]/len(y)*100:.1f}%)")
        
    def plot_correlation(x, y):
        fig, ax = plt.subplots(figsize=(10, 8))

        correlation_data = x.copy()
    
        if "Sex" in correlation_data.columns:
            correlation_data = pd.get_dummies(correlation_data, columns=["Sex"], drop_first=True, dtype=int)
        
        if "Embarked" in correlation_data.columns:
            correlation_data = pd.get_dummies(correlation_data, columns=["Embarked"], drop_first=True, dtype=int)
        
        correlation_data["Survived"] = y.values

        correlation_matrix = correlation_data.corr()

        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    center=0, square=True, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.show()
        
    def plot_boxplots(x, y):
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        viz_data = x.copy()
        viz_data["Survived"] = y.values
        viz_data["Outcome"] = viz_data["Survived"].map({1: "Survived", 0: "Died"})

        for idx, feature in enumerate(numeric_features):
            sns.boxplot(data=viz_data, x="Outcome", y=feature, ax=axes[idx], 
                        palette=["#2ecc71", "#e74c3c"])
            axes[idx].set_title(f"{feature} Distribution by Outcome", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Outcome", fontsize=11)
            axes[idx].set_ylabel(feature, fontsize=11)

        plt.tight_layout()
        plt.show()
        
class MoonsDataHelper():
    def load_dataset():
        moons_data = pd.read_csv("./input/moons.csv")
        x_train, x_test, y_train, y_test = split_train_test(moons_data.drop("Class_1", axis=1), moons_data["Class_1"])
        return x_train, x_test, y_train, y_test
    
    def plot_moons(x, y):
        x_array = x.values
        y_array = y.values
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_array[y_array==0, 0], x_array[y_array==0, 1], label='Class 0', alpha=0.6)
        plt.scatter(x_array[y_array==1, 0], x_array[y_array==1, 1], label='Class 1', alpha=0.6)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.title('Moon Dataset')
        plt.show()