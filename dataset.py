import pandas as pd

def main():
    df = pd.read_csv('cleaned_ids2018_sampled.csv', sep=',', usecols=[3, 28, 76, 21, 29, 78], nrows=20000)
    print(df.info())
    df['Label'] = df['Label'].replace({1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1})
    # balance
    rat = len(df.loc[df['Label'] == 0]) // len(df.loc[df['Label'] == 1])
    print(df['Label'].value_counts())
    df_1 = df.loc[df['Label'] == 1]
    df_1 = df_1.loc[df_1.index.repeat(rat)]
    df_n = pd.concat([df.loc[df['Label'] == 0], df_1]).sample(frac=1)
    print(df_n['Label'].value_counts())
    df_n.to_csv(path_or_buf="nir.csv")

if __name__ == '__main__':
    main()
