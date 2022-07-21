import os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, gc, os
# import cupy, cudf #
import fastparquet
import time

# https://www.kaggle.com/code/romaupgini/statement-dates-to-use-or-not-to-use
# hypothesis 4

# RANDOM SEED
SEED = 42
# FILL NAN VALUE
NAN_VALUE = -127


def hex_to_int_customer_ID(df):
    hex_to_int = lambda x: int(x, 16)
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    print('original ID', df['customer_ID'])
    df['customer_ID_int'] = data['customer_ID'].str[-16:].apply(hex_to_int).astype('int64')
    print('after hex_to_int ID: ', df['customer_ID_int'])
    df = df.drop(['customer_ID'], axis=1)
    df = df.rename(columns={'customer_ID_int': 'customer_ID'})

    print('shape of data:', df.shape)
    return df


def get_hypothesis4_feature(df):
    df.S_2 = pd.to_datetime(df.S_2)
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]

    # Initial feature selection to speed up fitting, based on @ambros
    # https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart/notebook
    features_avg = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14',
                    'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28',
                    'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42',
                    'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58',
                    'D_59', 'D_60', 'D_61', 'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74',
                    'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86', 'D_91', 'D_92', 'D_94', 'D_96',
                    'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120',
                    'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133',
                    'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1',
                    'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21',
                    'R_22', 'R_24', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16',
                    'S_18', 'S_22', 'S_23', 'S_25', 'S_26']
    features_min = ['B_2', 'B_4', 'B_5', 'B_9', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_20', 'B_28', 'B_33',
                    'B_36', 'B_42', 'D_39', 'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 'D_53', 'D_55',
                    'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_71', 'D_74', 'D_75', 'D_78', 'D_83', 'D_102',
                    'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 'D_132', 'D_140', 'D_141',
                    'D_144', 'D_145', 'P_2', 'P_3', 'R_1', 'R_27', 'S_3', 'S_5', 'S_7', 'S_11', 'S_12', 'S_23', 'S_25']
    features_max = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_12', 'B_13', 'B_14',
                    'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_21', 'B_23', 'B_24', 'B_25', 'B_30', 'B_33', 'B_37',
                    'B_38', 'B_39', 'B_40', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47',
                    'D_48', 'D_49', 'D_50', 'D_52', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_63', 'D_64',
                    'D_65', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84',
                    'D_91', 'D_102', 'D_105', 'D_107', 'D_110', 'D_111', 'D_112', 'D_115', 'D_116', 'D_117', 'D_118',
                    'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_131', 'D_132', 'D_133',
                    'D_134', 'D_135', 'D_136', 'D_138', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3',
                    'P_4', 'R_1', 'R_3', 'R_5', 'R_6', 'R_7', 'R_8', 'R_10', 'R_11', 'R_14', 'R_17', 'R_20', 'R_26',
                    'R_27', 'S_3', 'S_5', 'S_7', 'S_8', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_22', 'S_23', 'S_24',
                    'S_25', 'S_26', 'S_27']
    features_last = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13',
                     'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25',
                     'B_26', 'B_28', 'B_30', 'B_32', 'B_33', 'B_36', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42',
                     'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51',
                     'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_63', 'D_64',
                     'D_65', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_75', 'D_76', 'D_77', 'D_78', 'D_79', 'D_80',
                     'D_81', 'D_82', 'D_83', 'D_86', 'D_91', 'D_96', 'D_105', 'D_106', 'D_112', 'D_114', 'D_119',
                     'D_120', 'D_121', 'D_122', 'D_124', 'D_125', 'D_126', 'D_127', 'D_130', 'D_131', 'D_132', 'D_133',
                     'D_134', 'D_138', 'D_140', 'D_141', 'D_142', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3',
                     'R_4', 'R_5', 'R_6', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_12', 'R_13', 'R_14', 'R_15', 'R_19',
                     'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_11', 'S_12', 'S_13', 'S_16', 'S_19',
                     'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
    features_last = list(set(features_last) - set(cat_features))
    features_max = list(set(features_max) - set(cat_features))
    features_min = list(set(features_min) - set(cat_features))
    features_avg = list(set(features_avg) - set(cat_features))

    # Drop non stable features for train-test, based on % of NaNs
    # https://www.kaggle.com/code/onodera1/amex-eda-comparison-of-training-and-test-data
    # df.drop(["B_29", "S_9"], axis=1, inplace=True) # 이미 df에 빠져 있는 컬럼

    # Hypothesis #4 - retrieve info from statement dates as distance between the dates
    # Than calculate 'mean', 'std', 'max', 'last' statistics for distances
    # cudf doesn't support diff() as GroupBy function, slow pandas DF used
    temp = df[["customer_ID", "S_2"]]
    temp["SDist"] = temp.groupby("customer_ID")["S_2"].diff() / np.timedelta64(1, 'D')
    # Impute with average distance 30.53 days
    temp['SDist'].fillna(30.53, inplace=True)
    df = pd.concat([df, pd.DataFrame(temp["SDist"])], axis=1)
    del temp
    _ = gc.collect()
    features_last.append('SDist')
    features_avg.append('SDist')
    features_max.append('SDist')
    features_min.append('SDist')

    # https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514
    df.loc[(df.R_13 == 0) & (df.R_17 == 0) & (df.R_20 == 0) & (df.R_8 == 0), 'R_6'] = 0
    df.loc[df.B_39 == -1, 'B_36'] = 0

    # Compute "after pay" features
    # https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb
    # for bcol in [f'B_{i}' for i in [11, 14, 17]] + ['D_39', 'D_131'] + [f'S_{i}' for i in [16, 23]]:

    for bcol in [f'B_{i}' for i in [11, 14, 17]] + ['D_39', 'D_131'] + [f'S_{i}' for i in [16, 23]]:
        for pcol in ['P_2', 'P_3']:
            if bcol in df.columns:
                print('bcol', bcol, 'pcol', pcol)
                print('len', bcol, ':', sum(df[bcol].isna()), '   /    len', pcol, ':', sum(df[pcol].isna()))
                # df[[f'{bcol}-{pcol}']] = df[bcol] - df[pcol]
                df[f'{bcol}-{pcol}'] = df[bcol] - df[pcol]
                features_last.append(f'{bcol}-{pcol}')
                features_avg.append(f'{bcol}-{pcol}')
                features_max.append(f'{bcol}-{pcol}')
                features_min.append(f'{bcol}-{pcol}')

    # BASIC FEATURE ENGINEERING
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
    # https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb

    test_num_last = df.groupby("customer_ID")[features_last].agg(['last', 'first'])
    test_num_last.columns = ['_'.join(x) for x in test_num_last.columns]
    test_num_min = df.groupby("customer_ID")[features_min].agg(['min'])
    test_num_min.columns = ['_'.join(x) for x in test_num_min.columns]
    test_num_max = df.groupby("customer_ID")[features_max].agg(['max'])
    test_num_max.columns = ['_'.join(x) for x in test_num_max.columns]
    test_num_avg = df.groupby("customer_ID")[features_avg].agg(['mean'])
    test_num_avg.columns = ['_'.join(x) for x in test_num_avg.columns]
    test_num_std = df.groupby("customer_ID")[
        list(set().union(features_avg, features_last, features_min, features_max))].agg(['std', 'quantile'])
    test_num_std.columns = ['_'.join(x) for x in test_num_std.columns]

    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['last', 'first'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    # add last statement date, statements count and "new customer" category (LT=0.5)
    test_date_agg = df.groupby("customer_ID")[["S_2", "B_3", "D_104"]].agg(['last', 'count'])
    test_date_agg.columns = ['_'.join(x) for x in test_date_agg.columns]
    test_date_agg.rename(columns={'S_2_count': 'LT', 'S_2_last': 'S_2'}, inplace=True)
    test_date_agg.loc[(test_date_agg.B_3_last.isnull()) & (test_date_agg.LT == 1), 'LT'] = 0.5
    test_date_agg.loc[(test_date_agg.D_104_last.isnull()) & (test_date_agg.LT == 1), 'LT'] = 0.5
    test_date_agg.drop(["B_3_last", "D_104_last", "B_3_count", "D_104_count"], axis=1, inplace=True)

    df = pd.concat(
        [test_date_agg, test_num_last, test_num_min, test_num_max, test_num_avg, test_num_std, test_cat_agg], axis=1)
    del test_date_agg, test_num_last, test_num_min, test_num_max, test_num_avg, test_num_std, test_cat_agg

    # Ratios/diffs on last values as features, based on @ragnar123
    # https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977
    for col in list(set().union(features_last, features_avg)):
        try:
            df[f'{col}_last_first_div'] = df[f'{col}_last'] / df[f'{col}_first']
            df[f'{col}_last_mean_sub'] = df[f'{col}_last'] - df[f'{col}_mean']
            df[f'{col}_last_mean_div'] = df[f'{col}_last'] / df[f'{col}_mean']
            df[f'{col}_last_max_div'] = df[f'{col}_last'] / df[f'{col}_max']
            df[f'{col}_last_min_div'] = df[f'{col}_last'] / df[f'{col}_min']

        except:
            pass
    print('shape after engineering', df.shape)
    return df


if __name__ == "__main__":
    data = pd.read_feather(r"G:\내 드라이브\code\amex_default_predict\data\train_data.ftr")
    # 시간 측정
    start = time.time()  # 시작 시간 저장
    #######################################################
    # 작업 코드
    data = hex_to_int_customer_ID(data)
    train = get_hypothesis4_feature(data)
    #######################################################
    print("feature_engineer time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간 ; feature_engineer time : 1798.0522730350494

    # train.to_parquet('./features/hypothesis4_train.parquet', engine='fastparquet') # 오류 발생; RuntimeError: Compression 'snappy' not available.  Options: ['GZIP', 'UNCOMPRESSED']
    train.to_feather('./features/hypothesis4_train.ftr')
