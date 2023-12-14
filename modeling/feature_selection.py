from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping
import pandas as pd
import numpy as np

LIGHTGBM_PARAMS = {
    'boosting_type': 'goss',
    'n_estimators': 10000,
    'learning_rate': 0.005134,
    'num_leaves': 54,
    'max_depth': 10,
    'subsample_for_bin': 240000,
    'reg_alpha': 0.436193,
    'reg_lambda': 0.479169,
    'colsample_bytree': 0.508716,
    'min_split_gain': 0.024766,
    'subsample': 1,
    'is_unbalance': False,
    'silent':-1,
    'verbose':-1
}

def feature_importance(X_train, y_train):
    feature_importances = np.zeros(X_train.shape[1])
    model = LGBMClassifier(**{**LIGHTGBM_PARAMS})

    # Training the lgbm for 2 rounds to avoid overfitting
    for i in range(2):
        train_features, valid_features, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.25, random_state=i)
        model.fit(train_features, train_y, eval_set=[(valid_features, valid_y)], eval_metric='auc',
                  callbacks=[early_stopping(stopping_rounds=100)])

        feature_importances += model.feature_importances_

    # The final feature importance scores is the average of scores of 2 training rounds
    feature_importances /= 2
    feature_importances = pd.DataFrame({'feature': list(X_train.columns), 'importance': feature_importances}).sort_values('importance', 
                                                                                                                          ascending=False)
    # Return a list of features with zero importance, which would be dropped from the final data frame.
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    return zero_features