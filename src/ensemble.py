from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def get_ensemble_models():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Bagging": BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=10, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "Voting": VotingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('svc', make_pipeline(StandardScaler(), SVC(probability=True))),
            ('rf', RandomForestClassifier())
        ], voting='soft')
    }

    return models, X_train, X_test, y_train, y_test
