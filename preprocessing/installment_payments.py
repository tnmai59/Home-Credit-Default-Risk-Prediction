from utils import *
from configurations import INSTALLMENTS_AGG, INSTALLMENTS_TIME_AGG

def installment_payments_pipeline(path, num_rows= None):
    """ Process installments_payments.csv and return a pandas dataframe. """

    pay = pd.read_csv(os.path.join(path, 'dseb63_installments_payments.csv'), nrows= num_rows)
    # Group payments and get Payment difference
    pay = sum_agg(pay, ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], 'AMT_PAYMENT', 'AMT_PAYMENT_GROUPED')
    pay['PAYMENT_DIFFERENCE'] = pay['AMT_INSTALMENT'] - pay['AMT_PAYMENT_GROUPED']
    pay['PAYMENT_RATIO'] = pay['AMT_INSTALMENT'] / pay['AMT_PAYMENT_GROUPED']
    pay['PAID_OVER_AMOUNT'] = pay['AMT_PAYMENT'] - pay['AMT_INSTALMENT']
    pay['PAID_OVER'] = (pay['PAID_OVER_AMOUNT'] > 0).astype(int)
    # Payment Entry: Days past due and Days before due
    pay['DPD'] = pay['DAYS_ENTRY_PAYMENT'] - pay['DAYS_INSTALMENT']
    pay['DPD'] = pay['DPD'].apply(lambda x: 0 if x <= 0 else x)
    pay['DBD'] = pay['DAYS_INSTALMENT'] - pay['DAYS_ENTRY_PAYMENT']
    pay['DBD'] = pay['DBD'].apply(lambda x: 0 if x <= 0 else x)
    # Flag late payment
    pay['LATE_PAYMENT'] = pay['DBD'].apply(lambda x: 1 if x > 0 else 0)
    # Percentage of payments that were late
    pay['INSTALMENT_PAYMENT_RATIO'] = pay['AMT_PAYMENT'] / pay['AMT_INSTALMENT']
    pay['LATE_PAYMENT_RATIO'] = pay.apply(lambda x: x['INSTALMENT_PAYMENT_RATIO'] if x['LATE_PAYMENT'] == 1 else 0, axis=1)
    # Flag late payments that have a significant amount
    pay['SIGNIFICANT_LATE_PAYMENT'] = pay['LATE_PAYMENT_RATIO'].apply(lambda x: 1 if x > 0.05 else 0)
    # Flag k threshold late payments
    pay['DPD_7'] = pay['DPD'].apply(lambda x: 1 if x >= 7 else 0)
    pay['DPD_15'] = pay['DPD'].apply(lambda x: 1 if x >= 15 else 0)

    #----- BỔ SUNG-------------------------------------------------------------
    pay['INS_IS_DPD_UNDER_120'] = pay['DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
    pay['INS_IS_DPD_OVER_120'] = pay['DPD'].apply(lambda x: 1 if (x >= 120) else 0)

    #-----Bổ sung 12/4/2023-----------------------------------------------------
    # pay['DAYS_INSTALMENT_60'] = pay['DAYS_INSTALMENT'].apply(lambda x: 1 if (x>=60) else 0)
    # pay['DAYS_INSTALMENT_90'] = pay['DAYS_INSTALMENT'].apply(lambda x: 1 if (x>=90) else 0)
    # pay['DAYS_INSTALMENT_180'] = pay['DAYS_INSTALMENT'].apply(lambda x: 1 if (x>=180) else 0)
    # pay['DAYS_INSTALMENT_365'] = pay['DAYS_INSTALMENT'].apply(lambda x: 1 if (x>=365) else 0)
    #--------------------------------------------------------------------------
    # Aggregations by SK_ID_CURR
    pay_agg = group(pay, 'INS_', INSTALLMENTS_AGG)

    # Installments in the last x months
    for months in [36, 60]:
        recent_prev_id = pay[pay['DAYS_INSTALMENT'] >= -30*months]['SK_ID_PREV'].unique()
        pay_recent = pay[pay['SK_ID_PREV'].isin(recent_prev_id)]
        prefix = 'INS_{}M_'.format(months)
        pay_agg = group_and_merge(pay_recent, pay_agg, prefix, INSTALLMENTS_TIME_AGG)

    # Last x periods trend features
    group_features = ['SK_ID_CURR', 'SK_ID_PREV', 'DPD', 'LATE_PAYMENT',
                      'PAID_OVER_AMOUNT', 'PAID_OVER', 'DAYS_INSTALMENT']
    gp = pay[group_features].groupby('SK_ID_CURR')

    # Last loan features
    g = parallel_apply(gp, installments_last_loan_features, index_name='SK_ID_CURR', chunk_size=10000).reset_index()
    pay_agg = pay_agg.merge(g, on='SK_ID_CURR', how='left')
    return pay_agg


def installments_last_loan_features(gr):
    # Create a copy of the input groupby object
    gr_ = gr.copy()
    
    # Sort the group by 'DAYS_INSTALMENT' in descending order
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    
    # Get the SK_ID_PREV of the last installment
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    
    # Filter the group to include only the rows corresponding to the last installment
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

    # Initialize an empty dictionary to store computed features
    features = {}
    
    # Add features related to DPD (Days Past Due)
    features = add_features_in_group(features, gr_, 'DPD',
                                     ['sum', 'mean', 'max', 'std'],
                                     'LAST_LOAN_')
    
    # Add features related to LATE_PAYMENT
    features = add_features_in_group(features, gr_, 'LATE_PAYMENT',
                                     ['count', 'mean'],
                                     'LAST_LOAN_')
    
    # Add features related to PAID_OVER_AMOUNT
    features = add_features_in_group(features, gr_, 'PAID_OVER_AMOUNT',
                                     ['sum', 'mean', 'max', 'min', 'std'],
                                     'LAST_LOAN_')
    
    # Add features related to PAID_OVER
    features = add_features_in_group(features, gr_, 'PAID_OVER',
                                     ['count', 'mean'],
                                     'LAST_LOAN_')
    
    # Return the computed features
    return features
