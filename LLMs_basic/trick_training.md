# Early stopping 
+ LGBM:
        ```commandline
        Iteration: 590 	Log Loss: 1.0225806233972214
        Iteration: 600 	Log Loss: 1.0225275724859118
        Iteration: 610 	Log Loss: 1.0226533644541618
        Iteration: 620 	Log Loss: 1.0226788604754753
        ...
        Iteration: 680 	Log Loss: 1.0226227913025245
        Iteration: 690 	Log Loss: 1.0225613793431658
        Early stopping, best iteration is:
        [599]	valid_0's multi_logloss: 1.02247
        ```
+ max_estimators = 2000
+ early_stopping_limit = 100: meaning training will stop if the model performance does not improve for 100 consecutive rounds.

# Find max depth train LGBM using Grid Search
        ```commandline
        Find max depth ....
        Fitting 3 folds for each of 6 candidates, totalling 18 fits
        Best max_depth: 3
        Find max_depth cost:  7502.0411586761475
        ```
+ Code:
```doctest
    # Create the model
    model = lgb.LGBMClassifier(**params)
    # Define the grid of max_depth values to search
    param_grid = {'max_depth': [3, 4, 5, 6, 7, 8]}
    # Perform grid search
    print("Find max depth ....")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_log_loss', verbose=1)
    grid_search.fit(X_train, y_train)
```
