from utils import *
from category_encoders.target_encoder import TargetEncoder

def train_test_pipeline(path, num_rows = None):
    """ Process application_train.csv and application_test.csv and return a pandas dataframe. """
    train = pd.read_csv(os.path.join(path, 'dseb63_application_train.csv'), nrows= num_rows)
    test = pd.read_csv(os.path.join(path, 'dseb63_application_test.csv'), nrows= num_rows)

    # Create a list of feature from the training set exclude TARGET and SK_ID_CURR
    feats = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR']]
    # Encode all the categorical variables with TargetEncoder
    target = train['TARGET']
    enc = TargetEncoder(return_df = True)
    train_encode = enc.fit_transform(train[feats],target)
    train_encode['SK_ID_CURR'] = train['SK_ID_CURR']
    train_encode['TARGET'] = target
    test_encode = enc.transform(test[feats])
    test_encode['SK_ID_CURR'] = test['SK_ID_CURR']
    train = train_encode
    test = test_encode
    # df = train.append(test)
    df = pd.concat([train, test]).reset_index(drop=True)
    del train, test; gc.collect()

    # Data cleaning, replace all the invalid day by np.nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # Flag_document features - count and kurtosis
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)

    # Cut age into bins using feature 'DAYS_BIRTH'
    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))
    # Income by origin
    #Selecting the columns 'AMT_INCOME_TOTAL' and 'ORGANIZATION_TYPE' from the DataFrame then grouping the selected columns by 'ORGANIZATION_TYPE' and calculating the median income for each group
    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    # Mapping the calculated median income values to the original DataFrame based on the 'ORGANIZATION_TYPE'
    # This creates a new column 'NEW_INC_BY_ORG' with the median income corresponding to each organization type
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)

    # New features based on External sources
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    df['EXT_SOURCES_WEIGHTED_AVG'] =  (df['EXT_SOURCE_1']*2 + df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']*3)/3 # bổ sung

    # Ignore warnings related to encountering NaN values during calculations
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    # Iterate through different aggregation functions ('min', 'max', 'mean', 'median', 'var')
    for function_name in ['min', 'max', 'mean', 'median', 'var']:
        # Construct feature names based on the aggregation function applied
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        # Apply the specified aggregation function to the three 'EXT_SOURCE' features along the rows (axis=1)
        # and create a new feature in the DataFrame with the aggregated result
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    # Credit ratios
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # Income ratios
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    # Time ratios
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

    #---------------BỔ SUNG------------------------
    # Some other features created base on EXT_SOURCE 1, 2, 3
    df['EXT_SOURCE_1^2'] = df['EXT_SOURCE_1']**2
    df['EXT_SOURCE_2^2'] = df['EXT_SOURCE_2']**2
    df['EXT_SOURCE_3^2'] = df['EXT_SOURCE_3']**2
    df['EXT_SOURCE_1 EXT_SOURCE_2'] = df['EXT_SOURCE_1']*df['EXT_SOURCE_2']
    df['EXT_SOURCE_1 EXT_SOURCE_3'] = df['EXT_SOURCE_1']*df['EXT_SOURCE_3']
    df['EXT_SOURCE_2 EXT_SOURCE_3'] = df['EXT_SOURCE_2']*df['EXT_SOURCE_3']
    df['PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['APP_SCORE1_TO_FAM_CNT_RATIO'] = df['EXT_SOURCE_1'] / df['CNT_FAM_MEMBERS']
    df['APP_SCORE1_TO_GOODS_RATIO'] = df['EXT_SOURCE_1'] / df['AMT_GOODS_PRICE']
    df['APP_SCORE1_TO_CREDIT_RATIO'] = df['EXT_SOURCE_1'] / df['AMT_CREDIT']
    df['APP_SCORE1_TO_SCORE2_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_2']
    df['APP_SCORE1_TO_SCORE3_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_3']
    df['APP_SCORE2_TO_CREDIT_RATIO'] = df['EXT_SOURCE_2'] / df['AMT_CREDIT']
    df['APP_SCORE2_TO_CITY_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT_W_CITY']
    df['APP_SCORE2_TO_POP_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_POPULATION_RELATIVE']
    df['APP_SCORE2_TO_PHONE_CHANGE_RATIO'] = df['EXT_SOURCE_2'] / df['DAYS_LAST_PHONE_CHANGE']

    #---- Bổ sung 12/1/2023
    df['EXT_SOURCE_1 * DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] * df['DAYS_EMPLOYED']
    df['EXT_SOURCE_2 * DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] * df['DAYS_EMPLOYED']
    df['EXT_SOURCE_3 * DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] * df['DAYS_EMPLOYED']
    df['EXT_SOURCE_1 / DAYS_BIRTH'] = df['EXT_SOURCE_1'] / df['DAYS_BIRTH']
    df['EXT_SOURCE_2 / DAYS_BIRTH'] = df['EXT_SOURCE_2'] / df['DAYS_BIRTH']
    df['EXT_SOURCE_3 / DAYS_BIRTH'] = df['EXT_SOURCE_3'] / df['DAYS_BIRTH']

    #------------------------
     #features eng
    df['CHILDRE_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_CREDIT_PERCENTAGE'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']


    # Features related to children
    df['CNT_NON_CHILD'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    df['CHILD_TO_NON_CHILD_RATIO'] = df['CNT_CHILDREN'] / df['CNT_NON_CHILD']
    df['INCOME_PER_NON_CHILD'] = df['AMT_INCOME_TOTAL'] / df['CNT_NON_CHILD']
    df['CREDIT_PER_PERSON'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / (1 + df['CNT_CHILDREN'])
    df['CREDIT_PER_NON_CHILD'] = df['AMT_CREDIT'] / df['CNT_NON_CHILD']

    #age bins
    df['RETIREMENT_AGE'] = (df['DAYS_BIRTH'] < -14000).astype(int)
    df['DAYS_BIRTH_QCUT'] = pd.qcut(df['DAYS_BIRTH'], q=5, labels=False)

    #long employemnt
    df['LONG_EMPLOYMENT'] = (df['DAYS_EMPLOYED'] < -2000).astype(int)

    # create income band
    bins = [0, 30000, 65000, 95000, 130000, 160000, 190880, 220000, 275000, 325000, np.inf]
    labels = range(1, 11)
    df['INCOME_BAND'] = pd.cut(df['AMT_INCOME_TOTAL'], bins=bins, labels=labels, right=False)

    # flag asset
    df['FLAG_ASSET'] = np.nan
    filter_0 = (df['FLAG_OWN_CAR'] == 'N') & (df['FLAG_OWN_REALTY'] == 'N')
    filter_1 = (df['FLAG_OWN_CAR'] == 'Y') & (df['FLAG_OWN_REALTY'] == 'N')
    filter_2 = (df['FLAG_OWN_CAR'] == 'N') & (df['FLAG_OWN_REALTY'] == 'Y')
    filter_3 = (df['FLAG_OWN_CAR'] == 'Y') & (df['FLAG_OWN_REALTY'] == 'Y')

    df.loc[filter_0, 'FLAG_ASSET'] = 0
    df.loc[filter_1, 'FLAG_ASSET'] = 1
    df.loc[filter_2, 'FLAG_ASSET'] = 2
    df.loc[filter_3, 'FLAG_ASSET'] = 3

    # Groupby: Statistics for applications in the same group
    group = ['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE', 'CODE_GENDER']
    df = median_agg(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_MEDIAN')
    df = std_agg(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_STD')
    df = mean_agg(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_MEAN')
    df = std_agg(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_STD')
    df = mean_agg(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_MEAN')
    df = std_agg(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_STD')
    df = mean_agg(df, group, 'AMT_CREDIT', 'GROUP_CREDIT_MEAN')
    df = mean_agg(df, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_MEAN')
    df = std_agg(df, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_STD')

    # Encode categorical features (LabelEncoder)
    # df, le_encoded_cols = label_encoder(df, None)
    df = drop_application_columns(df)
    return df

def drop_application_columns(df):
    # Feature are dropped base on our EDA. Since we spot that these feature doesn't contribute much to the
    # prediction since the distribution on 2 classes are quite similar. Some of these features are very imbalance.
    drop_list = [
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR',
        'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
        'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
        'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
        'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'HOUSETYPE_MODE',
        'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE'
    ]
    # Drop most flag document columns
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df


def get_age_label(days_birth):
    # Return the age group label, output datatype is integer
    age_years = -days_birth / 365
    if age_years < 27: return 1
    elif age_years < 40: return 2
    elif age_years < 50: return 3
    elif age_years < 65: return 4
    elif age_years < 99: return 5
    else: return 0