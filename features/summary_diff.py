from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# preprocessing
def get_cat_num_columnname(train):
    features = train.drop(['customer_ID', 'S_2','target'], axis=1).columns.to_list()
    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_68', 'D_63', 'D_64', 'D_66']
    num_features = [col for col in features if col not in cat_features]
    return cat_features, num_features


# features
def get_summary_and_diff_train(train, train_labels):
    cat_features, num_features = get_cat_num_columnname(train)

    # summary
    ## numeric
    train_num_agg = train.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace = True)
    ## category
    train_cat_agg = train.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace = True)

    ## Transform float64 columns to float32
    cols = list(train_num_agg.dtypes[train_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        train_num_agg[col] = train_num_agg[col].astype(np.float32)
    ## Transform int64 columns to int32
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)

    # diff
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(train.groupby(['customer_ID'])):
        ## Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        ## Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    ## Concatenate
    df1 = np.concatenate(df1, axis=0)
    ## Transform to dataframe
    df1 = pd.DataFrame(df1, columns=[col + '_diff1' for col in df[num_features].columns])
    ## Add customer id
    df1['customer_ID'] = customer_ids

    # merge
    df = train_num_agg.merge(train_cat_agg, how='inner', on='customer_ID').merge(df1, how='inner',on='customer_ID').merge(train_labels, how='inner', on='customer_ID')
    return df

def get_summary_and_diff_test(test):
    cat_features, num_features = get_cat_num_columnname(train)

    test_num_agg = test.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
    test_num_agg.reset_index(inplace=True)
    test_cat_agg = test.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
    test_cat_agg.reset_index(inplace=True)
    # Transform float64 columns to float32
    cols = list(test_num_agg.dtypes[test_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        test_num_agg[col] = test_num_agg[col].astype(np.float32)
    # Transform int64 columns to int32
    cols = list(test_cat_agg.dtypes[test_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        test_cat_agg[col] = test_cat_agg[col].astype(np.int32)
    # Get the difference
    #test_diff = get_difference(test, num_features)
    # diff
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(test.groupby(['customer_ID'])):
        ## Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        ## Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    ## Concatenate
    df1 = np.concatenate(df1, axis=0)
    ## Transform to dataframe
    df1 = pd.DataFrame(df1, columns=[col + '_diff1' for col in df[num_features].columns])
    ## Add customer id
    df1['customer_ID'] = customer_ids

    df = test_num_agg.merge(test_cat_agg, how='inner', on='customer_ID').merge(df1, how='inner', on='customer_ID')
    return df


def encoding(train, test): #같은 값으로 encoding을 하기 위함
    cat_features, num_features = get_cat_num_columnname(train)

    cat_features_last = [f"{cf}_last" for cf in cat_features]
    for cat_col in cat_features_last:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
        test[cat_col] = encoder.transform(test[cat_col])
    return train, test



