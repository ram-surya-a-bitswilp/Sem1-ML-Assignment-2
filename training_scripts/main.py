from BreastCancerDataPreprocessor import breast_cancer_train_test_data
from training_scripts.train_models import train_all_classifiers


if __name__ == "__main__":
    (X_train, X_test, y_train, y_test) = breast_cancer_train_test_data()

    train_all_classifiers(X_train, X_test, y_train, y_test)