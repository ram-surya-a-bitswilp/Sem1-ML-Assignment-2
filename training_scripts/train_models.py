from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from model_utils import train_and_evaluate


def train_all_classifiers(X_train, X_test, y_train, y_test):
    """
    Instantiates and trains all 5 required model architectures.
    """
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "KNN_Classifier": KNeighborsClassifier(n_neighbors=5),
        "Gaussian_Naive_Bayes": GaussianNB(),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs= 2)
    }

    results = []
    for name, model in models.items():
        report = train_and_evaluate(model, name, X_train, y_train, X_test, y_test)
        results.append(report)

    return results