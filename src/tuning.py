from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def grid_search_tuning(model, params, X, y):
    grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_

def random_search_tuning(model, params, X, y, n_iter=100):
    random = RandomizedSearchCV(model, params, n_iter=n_iter, cv=5, n_jobs=-1)
    random.fit(X, y)
    return random.best_estimator_
