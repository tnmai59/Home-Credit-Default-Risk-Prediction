from utils import *
from configurations import CREDIT_CARD_AGG, CREDIT_CARD_TIME_AGG

def credit_card_pipeline(path, num_rows= None):
    """ Process credit_card_balance.csv and return a pandas dataframe. """
    cc = pd.read_csv(os.path.join(path, 'dseb63_credit_card_balance.csv'), nrows= num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=False)
    cc.rename(columns={'AMT_RECIVABLE': 'AMT_RECEIVABLE'}, inplace=True)
    # Amount used from limit
    cc['LIMIT_USE'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    # Current payment / Min payment
    cc['PAYMENT_DIV_MIN'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']
    # Late payment
    cc['LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    # How much drawing of limit
    cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    ###-------BỔ SUNG-------------------------------------
    cc['AMT_DRAWING_SUM'] = cc['AMT_DRAWINGS_ATM_CURRENT'] + cc['AMT_DRAWINGS_CURRENT'] + cc[
                                    'AMT_DRAWINGS_OTHER_CURRENT'] + cc['AMT_DRAWINGS_POS_CURRENT']
    cc['BALANCE_LIMIT_RATIO'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 0.00001)
    cc['CNT_DRAWING_SUM'] = cc['CNT_DRAWINGS_ATM_CURRENT'] + cc['CNT_DRAWINGS_CURRENT'] + cc[
                                        'CNT_DRAWINGS_OTHER_CURRENT'] + cc['CNT_DRAWINGS_POS_CURRENT'] + cc['CNT_INSTALMENT_MATURE_CUM']
    cc['MIN_PAYMENT_RATIO'] = cc['AMT_PAYMENT_CURRENT'] / (cc['AMT_INST_MIN_REGULARITY'] + 0.0001)
    cc['PAYMENT_MIN_DIFF'] = cc['AMT_PAYMENT_CURRENT'] - cc['AMT_INST_MIN_REGULARITY']
    cc['MIN_PAYMENT_TOTAL_RATIO'] = cc['AMT_PAYMENT_TOTAL_CURRENT'] / (cc['AMT_INST_MIN_REGULARITY'] +0.00001)
    cc['PAYMENT_MIN_DIFF'] = cc['AMT_PAYMENT_TOTAL_CURRENT'] - cc['AMT_INST_MIN_REGULARITY']
    cc['AMT_INTEREST_RECEIVABLE'] = cc['AMT_TOTAL_RECEIVABLE'] - cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['SK_DPD_RATIO'] = cc['SK_DPD'] / (cc['SK_DPD_DEF'] + 0.00001)


    #calculating the rolling Exponential Weighted Moving Average over months for certain features
    rolling_columns = [
        'AMT_BALANCE',
        'AMT_CREDIT_LIMIT_ACTUAL',
        'AMT_RECEIVABLE_PRINCIPAL',
        'AMT_RECEIVABLE',
        'AMT_TOTAL_RECEIVABLE',
        'AMT_DRAWING_SUM',
        'BALANCE_LIMIT_RATIO',
        'CNT_DRAWING_SUM',
        'MIN_PAYMENT_RATIO',
        'PAYMENT_MIN_DIFF',
        'MIN_PAYMENT_TOTAL_RATIO',
        'AMT_INTEREST_RECEIVABLE',
        'SK_DPD_RATIO' ]
    exp_weighted_columns = ['EXP_' + ele for ele in rolling_columns]
    cc[exp_weighted_columns] = cc.groupby(['SK_ID_CURR','SK_ID_PREV'])[rolling_columns].transform(lambda x: x.ewm(alpha = 0.7).mean())
    #---------------------------------------------
    # Aggregations by SK_ID_CURR
    cc_agg = cc.groupby('SK_ID_CURR').agg(CREDIT_CARD_AGG)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg.reset_index(inplace= True)

    # Last month balance of each credit card application
    last_ids = cc.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()
    last_months_df = cc[cc.index.isin(last_ids)]
    cc_agg = group_and_merge(last_months_df,cc_agg,'CC_LAST_', {'AMT_BALANCE': ['mean', 'max']})


    # Aggregations for last x months
    for months in [12, 24, 48]:
        cc_prev_id = cc[cc['MONTHS_BALANCE'] >= -months]['SK_ID_PREV'].unique()
        cc_recent = cc[cc['SK_ID_PREV'].isin(cc_prev_id)]
        prefix = 'INS_{}M_'.format(months)
        cc_agg = group_and_merge(cc_recent, cc_agg, prefix, CREDIT_CARD_TIME_AGG)
    return cc_agg
