import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

def grid_search_tuning(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Params: {grid.best_params_}")
    return grid.best_estimator_

def main():
    print("Loading data...")
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1],
            'solver': ['lbfgs', 'liblinear']
        }
    }

    results = []

    for name, model in models.items():
        print(f"\n🔧 Tuning {name}...")
        best_model = grid_search_tuning(model, param_grids[name], X_train, y_train)
        metrics = evaluate_model(best_model, X_test, y_test)
        print(f"✅ {name} Metrics: {metrics}")
        results.append({'Model': name, **metrics})

    print("\n🔀 Evaluating Ensemble Methods...")

    ensemble_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ]

    voting = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    voting.fit(X_train, y_train)
    voting_metrics = evaluate_model(voting, X_test, y_test)
    results.append({'Model': 'VotingClassifier', **voting_metrics})

    stacking = StackingClassifier(
        estimators=ensemble_estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    stacking.fit(X_train, y_train)
    stacking_metrics = evaluate_model(stacking, X_test, y_test)
    results.append({'Model': 'StackingClassifier', **stacking_metrics})

    print("\n📊 Final Evaluation Results:")
    df_results = pd.DataFrame(results)
    print(df_results.sort_values(by="accuracy", ascending=False))

if __name__ == "__main__":
    main()
