import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_model(X, y, pipeline, grid=None, scorer=None, rs=42):
    """
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        # stratify=y,
                                                        random_state=rs)

    # gscv = GridSearchCV(estimator=pipeline,
    #                     param_grid=grid,
    #                     scoring=scorer,
    #                     n_jobs=-1,
    #                     cv=5)

    # model = gscv.fit(X_train, y_train)

    model = pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        # 'fitted_gscv': model,
        # 'best_model': model.best_estimator_.named_steps['model'],
        # 'tuned_params': pd.Series(model.best_params_),
        # 'accuracy': accuracy_score(y_test, y_pred),
        # 'precision': precision_score(y_test, y_pred),
        # 'recall': recall_score(y_test, y_pred),
        # 'f1': f1_score(y_test, y_pred),
        # 'y_true': y_test,
        # 'y_predicted': pd.Series(data=y_pred, index=X_test.index)

        'model': model,
        'train_score': model.score(X_train, y_train),
        'test_score': model.score(X_test, y_test),
        'mean_absolute_error': np.mean(
            cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)),
        'mean_squared_error': np.mean(
            cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=3)),
        'r2_score': np.mean(
            cross_val_score(model, X_train, y_train, scoring='r2', cv=3))
    }

    return results
