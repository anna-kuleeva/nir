from locale import atof
import numpy as np
import pandas as pd
# from sklearn.linear_pca import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# from sklearn.pca_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

dtype = np.float64

fl_dur = np.empty(10000, np.float64)
tot_fw_pk = np.empty(10000, np.float64)
tot_bw_pk = np.empty(10000, np.float64)
tot_l_fw_pkt = np.empty(10000, np.float64)
fw_pkt_l_max = np.empty(10000, np.float64)
fw_pkt_l_min = np.empty(10000, np.float64)
fw_pkt_l_avg = np.empty(10000, np.float64)
fw_pkt_l_std = np.empty(10000, np.float64)
Bw_pkt_l_max = np.empty(10000, np.float64)
Bw_pkt_l_min = np.empty(10000, np.float64)
Bw_pkt_l_avg = np.empty(10000, np.float64)
Bw_pkt_l_std = np.empty(10000, np.float64)
fl_byt_s = np.empty(10000, np.float64)
fl_pkt_s = np.empty(10000, np.float64)
fl_iat_avg = np.empty(10000, np.float64)
fl_iat_std = np.empty(10000, np.float64)
fl_iat_max = np.empty(10000, np.float64)
fl_iat_min = np.empty(10000, np.float64)
fw_iat_tot = np.empty(10000, np.float64)
fw_iat_avg = np.empty(10000, np.float64)
fw_iat_std = np.empty(10000, np.float64)
fw_iat_max = np.empty(10000, np.float64)
fw_iat_min = np.empty(10000, np.float64)
bw_iat_tot = np.empty(10000, np.float64)
bw_iat_avg = np.empty(10000, np.float64)
bw_iat_std = np.empty(10000, np.float64)
bw_iat_max = np.empty(10000, np.float64)
bw_iat_min = np.empty(10000, np.float64)
fw_psh_flag = np.empty(10000, np.float64)
bw_psh_flag = np.empty(10000, np.float64)
fw_urg_flag = np.empty(10000, np.float64)
bw_urg_flag = np.empty(10000, np.float64)
fw_hdr_len = np.empty(10000, np.float64)
bw_hdr_len = np.empty(10000, np.float64)
fw_pkt_s = np.empty(10000, np.float64)
bw_pkt_s = np.empty(10000, np.float64)
pkt_len_min = np.empty(10000, np.float64)
pkt_len_max = np.empty(10000, np.float64)
pkt_len_avg = np.empty(10000, np.float64)
pkt_len_std = np.empty(10000, np.float64)
pkt_len_va = np.empty(10000, np.float64)
fin_cnt = np.empty(10000, np.float64)
syn_cnt = np.empty(10000, np.float64)
rst_cnt = np.empty(10000, np.float64)
pst_cnt = np.empty(10000, np.float64)
ack_cnt = np.empty(10000, np.float64)
urg_cnt = np.empty(10000, np.float64)
cwe_cnt = np.empty(10000, np.float64)
ece_cnt = np.empty(10000, np.float64)
down_up_ratio = np.empty(10000, np.float64)
pkt_size_avg = np.empty(10000, np.float64)
fw_seg_avg = np.empty(10000, np.float64)
bw_seg_avg = np.empty(10000, np.float64)
fw_byt_blk_avg = np.empty(10000, np.float64)
fw_pkt_blk_avg = np.empty(10000, np.float64)
fw_blk_rate_avg = np.empty(10000, np.float64)
bw_byt_blk_avg = np.empty(10000, np.float64)
bw_pkt_blk_avg = np.empty(10000, np.float64)
bw_blk_rate_avg = np.empty(10000, np.float64)
subfl_fw_pk = np.empty(10000, np.float64)
subfl_fw_byt = np.empty(10000, np.float64)
subfl_bw_pkt = np.empty(10000, np.float64)
subfl_bw_byt = np.empty(10000, np.float64)
fw_win_byt = np.empty(10000, np.float64)
bw_win_byt = np.empty(10000, np.float64)
Fw_act_pkt = np.empty(10000, np.float64)
fw_seg_min = np.empty(10000, np.float64)
atv_avg = np.empty(10000, np.float64)
atv_std = np.empty(10000, np.float64)
atv_max = np.empty(10000, np.float64)
atv_min = np.empty(10000, np.float64)
idl_avg = np.empty(10000, np.float64)
idl_std = np.empty(10000, np.float64)
idl_max = np.empty(10000, np.float64)
idl_min = np.empty(10000, np.float64)

def parse(tmp_str, i):  # parse all params
    help_list = tmp_str.split(',')
    help_list.reverse()
    # no need
    help_str = help_list.pop()
    help_str = help_list.pop()
    help_str = help_list.pop()

    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fl_dur[i] = help_tmp

    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    tot_fw_pk[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    tot_bw_pk[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    tot_l_fw_pkt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_pkt_l_max[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_pkt_l_min[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_pkt_l_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_pkt_l_std[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    Bw_pkt_l_max[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    Bw_pkt_l_min[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    Bw_pkt_l_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    Bw_pkt_l_std[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fl_byt_s[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fl_pkt_s[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fl_iat_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fl_iat_std[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fl_iat_max[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fl_iat_min[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_iat_tot[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_iat_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_iat_std[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_iat_max[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_iat_min[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_iat_tot[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_iat_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_iat_std[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_iat_max[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_iat_min[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_psh_flag[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_psh_flag[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_urg_flag[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_urg_flag[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_hdr_len[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_hdr_len[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_pkt_s[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_pkt_s[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    pkt_len_min[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    pkt_len_max[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    pkt_len_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    pkt_len_std[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    pkt_len_va[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fin_cnt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    syn_cnt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    rst_cnt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    pst_cnt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    ack_cnt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    urg_cnt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    cwe_cnt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    ece_cnt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    down_up_ratio[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    pkt_size_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_seg_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_seg_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_byt_blk_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_pkt_blk_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_blk_rate_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_byt_blk_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_pkt_blk_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_blk_rate_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    subfl_fw_pk[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    subfl_fw_byt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    subfl_bw_pkt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    subfl_bw_byt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_win_byt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    bw_win_byt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    Fw_act_pkt[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    fw_seg_min[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    atv_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    atv_std[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    atv_max[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    atv_min[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    idl_avg[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    idl_std[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    idl_max[i] = help_tmp
    help_str = help_list.pop()
    if help_str.__len__():
        help_tmp = atof(help_str)
    else:
        help_tmp = 0
    idl_min[i] = help_tmp
    help_str = help_list.pop()

def read_data():
    f = open("cleaned_ids2018_sampled.txt", "r")
    for i in range(10000):
        tmp_str = f.__next__()# read full line -- all info about transaction
        parse(tmp_str, i)
    f.close()

def myPCA():
    print("create matrix of components")
    matrix = np.vstack((fl_dur, tot_fw_pk, tot_bw_pk, tot_l_fw_pkt, fw_pkt_l_max, fw_pkt_l_min, fw_pkt_l_avg,
                        fw_pkt_l_std, Bw_pkt_l_max, Bw_pkt_l_min, Bw_pkt_l_avg, Bw_pkt_l_std, fl_byt_s, fl_pkt_s,
                        fl_iat_avg, fl_iat_std, fl_iat_max, fl_iat_min, fw_iat_tot, fw_iat_avg, fw_iat_std, fw_iat_max,
                        fw_iat_min, bw_iat_tot, bw_iat_avg, bw_iat_std, bw_iat_max, bw_iat_min, fw_psh_flag, bw_psh_flag,
                        fw_urg_flag, bw_urg_flag, fw_hdr_len, bw_hdr_len, fw_pkt_s, bw_pkt_s, pkt_len_min, pkt_len_max,
                        pkt_len_avg, pkt_len_std, pkt_len_va, fin_cnt, syn_cnt, rst_cnt, pst_cnt, ack_cnt, urg_cnt,
                        cwe_cnt, ece_cnt, down_up_ratio, pkt_size_avg, fw_seg_avg, bw_seg_avg, fw_byt_blk_avg,
                        fw_pkt_blk_avg, fw_blk_rate_avg, bw_byt_blk_avg, bw_pkt_blk_avg, bw_blk_rate_avg, subfl_fw_pk,
                        subfl_fw_byt, subfl_bw_pkt, subfl_bw_byt, fw_win_byt, bw_win_byt, Fw_act_pkt, fw_seg_min, atv_avg,
                        atv_std, atv_max, atv_min, idl_avg, idl_std, idl_max, idl_min))
    pca = PCA(n_components=6)
    pca.fit_transform(matrix.T)
    names = ['fl_dur', 'tot_fw_pk', 'tot_bw_pk', 'tot_l_fw_pkt', 'fw_pkt_l_max', 'fw_pkt_l_min', 'fw_pkt_l_avg',
                'fw_pkt_l_std', 'Bw_pkt_l_max', 'Bw_pkt_l_min', 'Bw_pkt_l_avg', 'Bw_pkt_l_std', 'fl_byt_s', 'fl_pkt_s',
                'fl_iat_avg', 'fl_iat_std', 'fl_iat_max', 'fl_iat_min', 'fw_iat_tot', 'fw_iat_avg', 'fw_iat_std', 'fw_iat_max',
                'fw_iat_min', 'bw_iat_tot', 'bw_iat_avg', 'bw_iat_std', 'bw_iat_max', 'bw_iat_min', 'fw_psh_flag', 'bw_psh_flag',
                'fw_urg_flag', 'bw_urg_flag', 'fw_hdr_len', 'bw_hdr_len', 'fw_pkt_s', 'bw_pkt_s', 'pkt_len_min', 'pkt_len_max',
                'pkt_len_avg', 'pkt_len_std', 'pkt_len_va', 'fin_cnt', 'syn_cnt', 'rst_cnt', 'pst_cnt', 'ack_cnt', 'urg_cnt',
                'cwe_cnt', 'ece_cnt', 'down_up_ratio', 'pkt_size_avg', 'fw_seg_avg', 'bw_seg_avg', 'fw_byt_blk_avg',
                'fw_pkt_blk_avg', 'fw_blk_rate_avg', 'bw_byt_blk_avg', 'bw_pkt_blk_avg', 'bw_blk_rate_avg', 'subfl_fw_pk',
                'subfl_fw_byt', 'subfl_bw_pkt', 'subfl_bw_byt', 'fw_win_byt', 'bw_win_byt', 'Fw_act_pkt', 'fw_seg_min', 'atv_avg',
                'atv_std', 'atv_max', 'atv_min', 'idl_avg', 'idl_std', 'idl_max', 'idl_min']
    print("Explained variance ratio: ", pca.explained_variance_ratio_)
    print("summ of information: ", pca.explained_variance_ratio_.cumsum() - 0.00924982)

    # number of components
    n_pcs = pca.components_.shape[0]

    # get the index of the most important feature on EACH component i.e. largest absolute value
    # using LIST COMPREHENSION HERE
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

    # get the names
    most_important_names = [names[most_important[i]] for i in range(n_pcs)]

    # using LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i + 1): most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    df = pd.DataFrame(sorted(dic.items()))
    print(df)

# if __name__ == '__main__':
print("start reading file...")
read_data()
print("start PCA...")
myPCA()
# print("summ of information: ", 0.58097396 + 0.1410123 + 0.12694644 + 0.08118217 + 0.04877775) # + 0.01362218 + 0.00401686 + 0.00308634) # эти параметры дали в сумме 0,999... инф
      # 1.31706172e-04 + 6.30744170e-05 + 5.56650941e-05 + 5.15205512e-05 + 2.06752570e-05 + 2.03773043e-05 + 1.39658694e-05 +
      # 1.21831587e-05 + 9.28139431e-06 + 2.02566895e-06 + 8.47351743e-07 + 2.43099292e-07)
