import numpy as np
import pandas as pd
from pathlib import Path
import copy
from sklearn.model_selection import train_test_split
import json
import datetime
import argparse


root_dir = Path('__file__').resolve().parent
raw_datasets_dir = root_dir / "data" / "raw"
preprocessd_datasets_dir = root_dir / "data" / "preprocessed"

data_name = "data.csv"
train_name = "train.csv"
test_name = "test.csv"
domain_name = "domain.json"
domain_info_name = "domain_info.txt"
raw_train_name = "raw_train.csv"
raw_test_name = "raw_test.csv"
all_raw_name = "all_raw.csv"
raw_name = "raw.csv"
raw_train_with_column_name = "raw_train_with_column.csv"

adult_columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]


def args_parse():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--dataset', type=str, help="used dataset [adult, small-adult, nume-adult, trafic, bitcoin, electricity, phoneme, jm, adding-adult, all] (default: all)", default="all")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

def make_discretizer(series, limit_bin):
    if type(series[0]) is str or type(series[0]) is np.bool_:
        value_set = sorted(list(set(series)))
        n_val = len(value_set)
        def discretizer(value):
            return value_set.index(value)
    else:
        max = series.max()
        min = series.min()
        if type(series[0]) in [int, np.int64, np.int32] and max-min <= limit_bin:
            def discretizer(value):
                return int(value-min)
            n_val = int(max-min) + 1
        else:
            def discretizer(value):
                return int(limit_bin * (value-min) / (max+0.1-min))
            n_val = int(limit_bin)
    return discretizer, n_val
    
def discretize(df, limit_bin):
    domain = {}
    df_ = copy.deepcopy(df)
    for col in df.columns:
        discretizer, n_val = make_discretizer(df[col], limit_bin)
        df_[col] = df[col].map(discretizer)
        domain[col] = n_val
    return df_, domain

def split_test(df, seed):
    df = df.sample(frac=1, random_state=seed)
    return train_test_split(df, test_size=0.1, random_state=seed)

def write_data(preprocessd_datasets_dir, dataset_name, df, limit_bin, seed):
    dataset_name_dir = preprocessd_datasets_dir / dataset_name
    dataset_name_dir.mkdir(parents=True, exist_ok=True)
    dis_df, domain = discretize(df, limit_bin)
    
    if dataset_name == "bitcoin":
        df['weight'] = df['weight'].map(lambda x: '{0:.6f}'.format(x))

    if dataset_name in [ "electricity" ]:
        df['date'] = df['date'].map(lambda x: '{0:.6f}'.format(x))
        df['nswprice'] = df['nswprice'].map(lambda x: '{0:.6f}'.format(x))
        df['nswdemand'] = df['nswdemand'].map(lambda x: '{0:.6f}'.format(x)) 
        df['vicprice'] = df['vicprice'].map(lambda x: '{0:.6f}'.format(x)) 
    
    # bitcoin data is too large and sample 500000 data
    if dataset_name == "bitcoin":
        sampled = df.sample(n=500000, random_state=seed)
        sampled.to_csv(dataset_name_dir / raw_name, index=None, sep=" ", header=False)
        df.to_csv(dataset_name_dir / all_raw_name, index=None, sep=" ", header=False)
        dis_df = dis_df.sample(n=500000, random_state=seed)
        dis_df.to_csv(dataset_name_dir / data_name, index=None)
    else:
        df.to_csv(dataset_name_dir / raw_name, index=None, sep=" ", header=False)
        dis_df.to_csv(dataset_name_dir / data_name, index=None)
    
    if dataset_name == "bitcoin":
        dis_df = dis_df.sample(n=110000, random_state=seed)
        df = df.sample(n=110000, random_state=seed)
        
    train, test = split_test(dis_df, seed)
    raw_train, raw_test = split_test(df, seed)
    raw_train.to_csv(dataset_name_dir / raw_train_name, index=None, sep=" ", header=False)
    
    if dataset_name == "adult":
        raw_train.columns = adult_columns

    raw_train.to_csv(dataset_name_dir / raw_train_with_column_name, index=None)
    raw_test.to_csv(dataset_name_dir / raw_test_name, index=None, sep=" ", header=False)
    train.to_csv(dataset_name_dir / train_name, index=None)
    test.to_csv(dataset_name_dir / test_name, index=None)
    
    with (dataset_name_dir / domain_name).open(mode="w") as f:
            json.dump(domain, f)
    print(domain)

            
if __name__ == '__main__':
    args = args_parse()
    if args.dataset == "all" or args.dataset == "adult":
        dataset_dir = raw_datasets_dir / "adult"
        df = pd.read_csv(dataset_dir / "adult.data", header=None)
        test_df = pd.read_csv(dataset_dir / "adult.test", header=None)
        df = pd.concat([df, test_df], ignore_index=True)
        df[1] = df[1].str.strip()
        df[3] = df[3].str.strip()
        df[5] = df[5].str.strip()
        df[6] = df[6].str.strip()
        df[7] = df[7].str.strip()
        df[8] = df[8].str.strip()
        df[9] = df[9].str.strip()
        df[13] = df[13].str.strip()
        df[14] = df[14].str.strip()

        df[14].loc[df[14] == "<=50K."] = "<=50K"
        df[14].loc[df[14] == ">50K."] = ">50K"

        write_data(preprocessd_datasets_dir, 'adult', df, 100, args.seed)


        domain_info ="""\
C 17.0 90.0
D Federal-gov Local-gov State-gov Private Self-emp-inc Self-emp-not-inc Without-pay Never-worked ?
C 12285.0 1490400.0
D Preschool 1st-4th 5th-6th 7th-8th 9th 10th 11th 12th HS-grad Some-college Assoc-voc Assoc-acdm Bachelors Masters Prof-school Doctorate
C 1.0 16.0
D Never-married Married-AF-spouse Married-civ-spouse Married-spouse-absent Separated Widowed Divorced
D Adm-clerical Armed-Forces Craft-repair Exec-managerial Farming-fishing Handlers-cleaners Machine-op-inspct Other-service Priv-house-serv Prof-specialty Protective-serv Sales Tech-support Transport-moving ?
D Husband Wife Own-child Other-relative Not-in-family Unmarried
D White Black Amer-Indian-Eskimo Asian-Pac-Islander Other
D Female Male
C 0.0 99999.0
C 0.0 4356.0
C 1.0 99.0
D United-States Cambodia England Puerto-Rico Canada Germany Outlying-US(Guam-USVI-etc) India Japan Greece South China Cuba Iran Honduras Philippines Italy Poland Jamaica Vietnam Mexico Portugal Ireland France Dominican-Republic Laos Ecuador Taiwan Haiti Columbia Hungary Guatemala Nicaragua Scotland Thailand Yugoslavia El-Salvador Trinadad&Tobago Peru Hong Holand-Netherlands ?
D <=50K >50K
"""

        with open(preprocessd_datasets_dir / "adult" / domain_info_name, mode='w') as f:
            f.write(domain_info)

    if args.dataset == "all" or args.dataset == "small-adult":
        dataset_dir = raw_datasets_dir / "adult"
        df = pd.read_csv(dataset_dir / "adult.data", header=None)
        test_df = pd.read_csv(dataset_dir / "adult.test", header=None)
        df = pd.concat([df, test_df], ignore_index=True)
        df[14] = df[14].str.strip()
        df[14].loc[df[14] == "<=50K."] = "<=50K"
        df[14].loc[df[14] == ">50K."] = ">50K"
        df = df.drop(columns=[2,3,4,5,6,7,9,11,12,13,14])

        df[1] = df[1].str.strip()
        df[8] = df[8].str.strip()

        domain_info ="""\
C 17.0 90.0
D Federal-gov Local-gov State-gov Private Self-emp-inc Self-emp-not-inc Without-pay Never-worked ?
D White Black Amer-Indian-Eskimo Asian-Pac-Islander Other
C 0.0 99999.0
"""

        write_data(preprocessd_datasets_dir, 'small-adult', df, 100, args.seed)

        with open(preprocessd_datasets_dir / "small-adult" / domain_info_name, mode='w') as f:
            f.write(domain_info)

    if args.dataset == "all" or args.dataset == "nume-adult":
        dataset_dir = raw_datasets_dir / "adult"
        df = pd.read_csv(dataset_dir / "adult.data", header=None)
        test_df = pd.read_csv(dataset_dir / "adult.test", header=None)
        df = pd.concat([df, test_df], ignore_index=True)
        df[14] = df[14].str.strip()
        df[14].loc[df[14] == "<=50K."] = "<=50K"
        df[14].loc[df[14] == ">50K."] = ">50K"
        df = df.drop(columns=[1,3,5,6,7,8,9,13])

        write_data(preprocessd_datasets_dir, 'nume-adult', df, 100, args.seed)


        domain_info ="""\
C 17.0 90.0
C 12285.0 1490400.0
C 1.0 16.0
C 0.0 99999.0
C 0.0 4356.0
C 1.0 99.0
D <=50K >50K
"""

        with open(preprocessd_datasets_dir / "nume-adult" / domain_info_name, mode='w') as f:
            f.write(domain_info)    

    if args.dataset == "all" or args.dataset == "bitcoin":
        dataset_dir = raw_datasets_dir / "bitcoin"
        df = pd.read_csv(dataset_dir / "BitcoinHeistData.csv")
        df = df.drop(columns=['address'])
        df['label'][df['label'] != "white"] = "black"


        write_data(preprocessd_datasets_dir, 'bitcoin', df, 30, args.seed)

        domain_info ="""\
C 2011.0 2018.0
C 1.0 365.0
C 0.0 144.0
C 0.0 1944.0
C 1.0 14497.0
C 0.0 14496.0
C 1.0 12920.0
C 30000000.0 49964398238996.0
D white black
"""

        with open(preprocessd_datasets_dir / "bitcoin" / domain_info_name, mode='w') as f:
            f.write(domain_info)


    if args.dataset == "all" or args.dataset == "electricity":
        dataset_dir = raw_datasets_dir / "electricity"
        df = pd.read_csv(dataset_dir / "electricity-normalized.csv")
        df = df.drop(columns=['day'])
        translate_table = {value: i for i, value in enumerate(np.sort(df['period'].unique()))}
        df['period'] = df['period'].map(translate_table)
        write_data(preprocessd_datasets_dir, 'electricity', df, 100, args.seed)

        domain_info ="""\
C 0.0 1.0
C 0.0 47.0
C 0.0 1.0
C 0.0 1.0
C 0.0 1.0
C 0.0 1.0
C 0.0 1.0
D UP DOWN
"""

        with open(preprocessd_datasets_dir / "electricity" / domain_info_name, mode='w') as f:
            f.write(domain_info)

    if args.dataset == "all" or args.dataset == "phoneme":
        dataset_dir = raw_datasets_dir / "phoneme"
        df = pd.read_csv(dataset_dir / "php8Mz7BG.csv")

        write_data(preprocessd_datasets_dir, 'phoneme', df, 10, args.seed)

        domain_info ="""\
C -2.94 3.83
C -3.1 3.7
C -2.8 2.7
C -2.5 3.1
C -2.4 4.6
D 1 2
"""

        with open(preprocessd_datasets_dir / "phoneme" / domain_info_name, mode='w') as f:
            f.write(domain_info)
    

    if args.dataset == "all" or args.dataset == "trafic":
        dataset_dir = raw_datasets_dir / 'trafic'
        df = pd.read_csv(dataset_dir / 'Metro_Interstate_Traffic_Volume.csv')
        df = df.drop(columns=['weather_description'])
        df = df.dropna()
        df['date_time'] = df['date_time'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())

        df['holiday'] = df['holiday'].map(lambda x: x.replace(' ', '-'))

        write_data(preprocessd_datasets_dir, 'trafic', df, 100, args.seed)

        domain_info =f"""\
D {' '.join(list(df['holiday'].unique()))}
C 0.0 310.07
C 0.0 9831.3
C 0.0 0.51
C 0 100
D {' '.join(list(df['weather_main'].unique()))}
C 1349168400.0 1538348400.0
C 0 7280
"""

        with open(preprocessd_datasets_dir / "trafic" / domain_info_name, mode='w') as f:
            f.write(domain_info)


    if args.dataset == "all" or args.dataset == "jm":
        dataset_dir = raw_datasets_dir / 'jm'
        df = pd.read_csv(dataset_dir / 'jm1.csv')
        df = df.dropna()

        df = df[df['uniq_Op'] != '?']
        df = df[df['uniq_Opnd'] != '?']
        df = df[df['total_Op'] != '?']
        df = df[df['total_Opnd'] != '?']
        df = df[df['branchCount'] != '?']

        df['uniq_Op'] = df['uniq_Op'].map(lambda x: float(x))
        df['uniq_Opnd'] = df['uniq_Opnd'].map(lambda x: float(x))
        df['total_Op'] = df['total_Op'].map(lambda x: float(x))
        df['total_Opnd'] = df['total_Opnd'].map(lambda x: float(x))
        df['branchCount'] = df['branchCount'].map(lambda x: float(x))

        write_data(preprocessd_datasets_dir, 'jm', df, 10, args.seed)

        domain_info =f"""\
C 1.0 3442.0
C 1.0 470.0
C 1.0 165.0
C 1.0 402.0
C 0.0 8441.0
C 0.0 80843.08
C 0.0 1.3
C 0.0 418.2
C 0.0 569.78
C 0.0 31079782.27
C 0.0 26.95
C 0.0 1726654.57
C 0 2824
C 0 344
C 0 447
C 0 108
C 0.0 411.0
C 0.0 1026.0
C 0.0 5420.0
C 0.0 3021.0
C 1.0 826.0
D False True
"""

        with open(preprocessd_datasets_dir / "jm" / domain_info_name, mode='w') as f:
            f.write(domain_info)

            
    if args.dataset == "all" or args.dataset == "gowalla":
        dataset_dir = raw_datasets_dir / 'gowalla'
        df = pd.read_csv(dataset_dir / "loc-gowalla_totalCheckins.txt", header=None, sep='\t')
        df = df.drop(columns=[0])
        df.columns = ['time', 'lat', 'lng', 'location_id']
        df = df[df['lat'] <= 90]
        df['time'] = df['time'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp())
        # write_data(data_dir, 'gowalla-4d', df, 100)

        df = df.drop(columns=['location_id'])
        # write_data(data_dir, 'gowalla-3d', df, 1000)

        df = df.drop(columns=['time'])
        write_data(preprocessd_datasets_dir, 'gowalla-2d', df, 1000, args.seed)