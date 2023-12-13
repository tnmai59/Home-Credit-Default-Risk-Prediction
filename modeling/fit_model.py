from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from feature_selection import *
from utils import timer, handling_outliers_3sigma, find_features
import re
import sys

def fit_model(df):
    with timer('Drop unnamed columns'):
      unnamed_columns = df.filter(like='Unnamed').columns.tolist()
      df.drop(columns=unnamed_columns, inplace=True)
      print('Shape after drop unnamed columns: {df.shape}')

    with timer('Replace infinity by nan'):
      df.replace([np.inf, -np.inf], np.nan, inplace=True)

    with timer('Rename features'):
      df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    with timer("Handling outliers"):
      cols = find_features(df)
      for col in cols:
        print(col)
        df[col] = handling_outliers_3sigma(col, df, df)
        print('------------------------\n')

    with timer('Split train test set'):
      train = df[df['TARGET'].notnull()]
      test = df[df['TARGET'].isnull()]
      X_train = train.drop(columns=['SK_ID_CURR', 'TARGET'])
      y_train = train['TARGET']
      X_test = test.drop(columns=['SK_ID_CURR', 'TARGET'])

    with timer("Feature selection"):
      to_drop = feature_importance(X_train, y_train)
      X_train.drop(columns = to_drop, inplace = True)
      X_test.drop(columns = to_drop, inplace = True)
      print(f'X_train shape: {X_train.shape}')
      print(f'X_test shape: {X_test.shape}')

    with timer('Fill nan'):
      imputer = SimpleImputer(strategy='median')
      imputer = imputer.fit(X_train)
      X_train = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
      X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    with timer('Scale by StandardScaler'):
      std = StandardScaler()
      X_train = pd.DataFrame(std.fit_transform(X_train), columns=X_train.columns)
      std2 = StandardScaler()
      X_test = pd.DataFrame(std2.fit_transform(X_test), columns=X_test.columns)

    with timer('Start tune with GridSearchCV'):
      model = LogisticRegression()
      param_grid = {
            'penalty' : [
                         'l2'
                        #  'l1',
#                          'elasticnet'
                         ],
            'C' :[
                # 0.001,
                0.01,
                  # 0.1, 1,
                  # 10
                # 15, 20, 25
                ],
            'solver' : [
                'liblinear',
                # 'saga'
                # 'lbfgs',
                # 'newton-cg',
#                 'sag'
            ],
            # 'max_iter' : [700]
        }
      
      grid_search = GridSearchCV(estimator=model, param_grid=param_grid,verbose=10,scoring='roc_auc_ovr')
      grid_search.fit(X_train, y_train)
      print(f'Best param: {grid_search.best_params_}')
      y_prob_train = grid_search.predict_proba(X_train)[:,1]
      y_prob_test = grid_search.predict_proba(X_test)[:,1]
      print(f'AUC: {roc_auc_score(y_train, y_prob_train)}')

    with timer("Export submission"):
      submit = test[['SK_ID_CURR']]
      submit.loc[:, 'TARGET'] = y_prob_test
      submit.to_csv('D:/Data Preparation final project group 2/outputs/submission.csv', index=False)
      print(submit)


if __name__ == "__main__":
#     print(sys.path)
#     sys.path.append('d:\\Data Preparation final project group 2')
    with timer("Pipeline total time"):
        df = pd.read_csv('D:/Data Preparation final project group 2/Data/final_data.csv')
        fit_model(df)
