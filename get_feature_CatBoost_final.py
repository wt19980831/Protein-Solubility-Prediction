import Bio.SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis                  # 用于计算蛋白质的一些特性
from propy import PseudoAAC                                         # 用于计算蛋白质的一些特性
import numpy as np
from multiprocessing import Pool ,cpu_count
import networkx as nx
import math
from Bio import SeqIO
from propy import CTD


def read_seq(path):
    res = []
    lines = open(path ,"r").readlines()
    for line in lines:
        res.append(line.strip())
    return res

def read_bio(path):
    res = []
    lines = open(path ,"r").readlines()
    for line in lines:
        res.append([float(x) for x in line.strip().split("\t")[0:-1]])
    return res

def read_tgt(path):
    res = []
    lines = open(path ,"r").readlines()
    for line in lines:
        res.append(float(line.strip()))
    return res

std = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

EIIP={
    "A": 0.0373,
    "C": 0.0829,
    "D": 0.1263,
    "E": 0.0058,
    "F": 0.0946,
    "G": 0.005,
    "H": 0.0242,
    "I": 0,
    "K": 0.0371,
    "L": 0,
    "M": 0.0823,
    "N": 0.0036,
    "P": 0.0198,
    "Q": 0.0761,
    "R": 0.0959,
    "S": 0.0829,
    "T": 0.0941,
    "V": 0.0057,
    "W": 0.0548,
    "Y": 0.0516 }

KYTE={
    "R": -4.5,
    "K": -3.9,
    "N": -3.5,
    "D": -3.5,
    "Q": -3.5,
    "E": -3.5,
    "H": -3.2,
    "P": -1.6,
    "Y": -1.3,
    "W": -0.9,
    "S": -0.8,
    "T": -0.7,
    "G": -0.4,
    "A": 1.8,
    "M": 1.9,
    "C": 2.5,
    "F": 2.8,
    "L": 3.8,
    "V": 4.2,
    "I": 4.5 }

mole_weight={
            "A": 89.09,
            "C": 121.16,
            "D": 133.10,
            "E": 147.13,
            "F": 165.19,
            "G": 75.07,
            "H": 155.16,
            "I": 131.17,
            "K": 146.19,
            "L": 131.17,
            "M": 149.21,
            "N": 132.12,
            "P": 115.13,
            "Q": 146.15,
            "R": 174.20,
            "S": 105.09,
            "T": 119.16,
            "V": 117.15,
            "W": 204.22,
            "Y": 181.19 }

sp_dict={
        "A":71.07,
        "C":103.10,
        "D":115.08,
        "E":129.11,
        "F":147.17,
        "G":57.05,
        "H":137.14,
        "I":113.15,
        "K":128.17,
        "L":113.15,
        "M":131.19,
        "N":114.10,
        "P":97.11,
        "Q":128.13,
        "R":156.18,
        "S":87.07,
        "T":101.14,
        "V":99.13,
        "W":186.20,
        "Y":163.17 }

'''
非极性       [0,0,0,1]
极性不带电    [0,0,1,0]
极性带负电    [0,1,0,0]
极性带正电    [0,1,0,0]
'''
ploar_dict = {
            "A":[0,0,0,1],
            "C":[0,0,1,0],
            "D":[0,1,0,0],
            "E":[0,1,0,0],
            "F":[0,0,0,1],
            "G":[0,0,1,0],
            "H":[0,1,0,0],
            "I":[0,0,0,1],
            "K":[0,1,0,0],
            "L":[0,0,0,1],
            "M":[0,0,0,1],
            "N":[0,0,1,0],
            "P":[0,0,0,1],
            "Q":[0,0,1,0],
            "R":[0,1,0,0],
            "S":[0,0,1,0],
            "T":[0,0,1,0],
            "V":[0,0,0,1],
            "W":[0,0,0,1],
            "Y":[0,0,1,0] }

'''
酸性：
中性：[0,0,1]
酸性：[0,1,0]
碱性：[1,0,0]
'''
acid_dict = {
            "A":[0,0,1],
            "C":[0,0,1],
            "D":[0,1,0],
            "E":[0,1,0],
            "F":[0,0,1],
            "G":[0,0,1],
            "H":[1,0,0],
            "I":[0,0,1],
            "K":[1,0,0],
            "L":[0,0,1],
            "M":[0,0,1],
            "N":[0,0,1],
            "P":[0,0,1],
            "Q":[0,0,1],
            "R":[1,0,0],
            "S":[0,0,1],
            "T":[0,0,1],
            "V":[0,0,1],
            "W":[0,0,1],
            "Y":[0,0,1] }

def graph(seq):
    g = nx.Graph()
    edges=[x+y for x in std for y in std]
    weights = {edge:0 for edge in edges}

    i = 0
    k = 1
    while i+k<len(seq):
        n0 = seq[i]
        n1 = seq[i+1]
        weights[n0+n1] = weights[n0+n1]+1
        i = i+k

    for k ,v in weights.items():
        k0 = list(k)[0]
        k1 = list(k)[1]
        if k0!=k1:
            g.add_edge(k0 ,k1 ,weight=v)
            g.add_edge(k1 ,k0 ,weight=v)
    res1 = [x[1] for x in g.degree()]
    res2 = list(nx.algorithms.degree_centrality(g).values())
    res3 = list(nx.clustering(g).values())
    return res1 + res2 + res3

def onehot(seq ,n=50):
    res = np.zeros((n ,20))
    for i,x in enumerate(seq[0:n]):
        res[i,std.index(x)] = 1
    return res.flatten().tolist()

def kyte(seq):
    res = np.array([KYTE[x] for x in seq])
    h ,e = np.histogram(res ,bins=64 ,density=True)
    res = h.tolist()
    return res

def ACC(seq):
    res = []
    for x in std:
        t = seq.count(x)
        res.append(t)
    return res

def ACC2(seq):
    xx = [x+y for x in std for y in std]
    res = [seq.count(x) for x in xx]
    h ,e = np.histogram(res ,bins=64)
    res = h.tolist()
    return res

def ACC3(seq):
    xx = [x+y+z for x in std for y in std for z in std]
    res = [seq.count(x) for x in xx]
    h ,e = np.histogram(res ,bins=64)
    res = h.tolist()
    return res

# def ACC4(seq):
#     xx = [x+y+z+p for x in std for y in std for z in std for p in std]
#     res = [seq.count(x) for x in xx]
#     h ,e = np.histogram(res ,bins=64)
#     res = h.tolist()
#     return res

def eiip(seq):
    res = np.array([EIIP[x] for x in seq])
    h ,e = np.histogram(res ,bins=64)
    res = h.tolist()
    return res

# def ctd(seq):
#     res = list(CTD.CalculateCTD(seq).values())
#     h ,e = np.histogram(np.array(res) ,bins=64)
#     res = h.tolist()
#     return res

def mole_w(seq):
    res = np.array([mole_weight[x] for x in seq])
    h ,e = np.histogram(res ,bins=64)
    res = h.tolist()
    return res

def sp(seq):
    res = np.array([sp_dict[x] for x in seq])
    h ,e = np.histogram(res ,bins=64)
    res = h.tolist()
    return res

def polar(seq ,n=10):
    sub_seq = seq[0:n]
    res = [[0,0,0,0]]*n
    for i ,x in enumerate(sub_seq):
        res[i]=ploar_dict[x]
    return np.array(res).flatten().tolist()

def acid(seq ,n=10):
    sub_seq = seq[0:n]
    res = [[0,0,0]]*n
    for i ,x in enumerate(sub_seq):
        res[i]=acid_dict[x]
    return np.array(res).flatten().tolist()

from gensim.models import Word2Vec
model_w2c = Word2Vec.load("word2vec.model")

def onehotw(seq ,n=50):
    sub_seq = seq[0:n]
    res = [0] * n
    for i ,x in enumerate(sub_seq):
        res[i] = model_w2c.wv[x][0]
    return res

# def GAAC(seq):
#     G1 = ['G','A','V','L','M','I']
#     G2 = ['F','Y','W']
#     G3 = ['K','R','H']
#     G4 = ['D','E']
#     G5 = ['S','T','C','P','N','Q']
#     ref = [G1,G2,G3,G4,G5]
#     enc = [0,0,0,0,0]
#     for ele in seq:
#         for i in range(5):
#             if ele in ref[i]:
#                 enc[i] += 1
#             else:
#                 continue
#     i = 0
#     for ele in enc:
#         enc[i] = ele/len(seq)
#         i+=1
#     return enc
#
# def RAAC(feature):
#     G1 = ['R', 'D', 'E', 'N', 'Q', 'K', 'H']    # Strongly_hydrophilic
#     G2 = ['L', 'I', 'A', 'V', 'M', 'F']         # Strongly_hydrophobic
#     G3 = ['S', 'T', 'Y', 'W']
#     G4 = ['P']                                  # Proline
#     G5 = ['G']                                  # Glycine
#     G6 = ['C']                                  # Cysteine
#     ref = [G1,G2,G3,G4,G5,G6]
#     seq = []
#     enc = np.zeros([6, 6])
#     for i in range(len(feature)-1):
#        char = feature[i]+feature[i+1]
#        seq.append(char)
#        i+=1
#     for ele in seq:
#         for i in range(6):
#             if ele[0] in ref[i]:
#                 for j in range(6):
#                     if ele[1] in ref[j]:
#                         enc[i][j] +=1
#                         continue
#                     else:continue
#                     j+=1
#             else:
#                 continue
#             i+=1
#     enc_r=[]
#     for i in range(6):
#         for j in range(6):
#             if enc[i][j]!=0:
#                 enc_r.append(enc[i][j]/(len(feature)-1))
#             else:
#                 enc_r.append(0)
#                 continue
#             j+=1
#         i+=1
#     return enc_r

def get_features(seq):
    '''
    CTD 有害
    ACC4 有害
    '''
    n = 10                                  # 序列截断长度为 10
    x1 = onehot(seq ,n=n)                   # 200 维度
    x2 = graph(seq)                         # 60 维
    x3 = kyte(seq)                          # 64 维
    x4 = eiip(seq)                          # 64 维
    x5 = ACC(seq) + ACC2(seq) + ACC3(seq)   # 20 + 64 + 64 = 148 维
    x6 = mole_w(seq)                        # 64 维
    x7 = sp(seq)                            # 64 维
    x8 = polar(seq ,n=n)                    # 40 维
    x9 = acid(seq ,n=n)                     # 30 维
    x10 = onehotw(seq ,n=n)                 # 10 维
    return  x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10


pool = Pool(cpu_count())

train_seq_path = "data/train_src.txt"
train_bio_path = "data/train_src_bio.txt"
train_tgt_path = "data/train_tgt.txt"

val_seq_path = "data/val_src.txt"
val_bio_path = "data/val_src_bio.txt"
val_tgt_path = "data/val_tgt.txt"

test_seq_path = "data/test_src.txt"
test_bio_path = "data/test_src_bio.txt"
test_tgt_path = "data/test_tgt.txt"


train_seq_fs = pool.map(get_features ,read_seq(train_seq_path))
val_seq_fs = pool.map(get_features ,read_seq(val_seq_path))
test_seq_fs = pool.map(get_features ,read_seq(test_seq_path))

train_bio_fs = read_bio(train_bio_path)
val_bio_fs = read_bio(val_bio_path)
test_bio_fs = read_bio(test_bio_path)

train_label = read_tgt(train_tgt_path)
val_label = read_tgt(val_tgt_path)
test_label = read_tgt(test_tgt_path)

train_label = np.array(train_label)
val_label = np.array(val_label)
test_label = np.array(test_label)

print(len(train_seq_fs) ,len(train_bio_fs) ,len(train_label))
print(len(val_seq_fs) ,len(val_bio_fs) ,len(val_label))
print(len(test_seq_fs) ,len(test_bio_fs) ,len(test_label))

train_fs = [train_seq_fs[i] + train_bio_fs[i] for i in range(len(train_seq_fs))]
val_fs = [val_seq_fs[i] + val_bio_fs[i] for i in range(len(val_seq_fs))]
test_fs = [test_seq_fs[i] + test_bio_fs[i] for i in range(len(test_seq_fs))]


train_data = np.array(train_fs)
val_data = np.array(val_fs)
test_data = np.array(test_fs)

tt_data = np.concatenate([train_data ,test_data])
tt_label = np.concatenate([train_label ,test_label])

tv_data = np.concatenate([train_data ,val_data])
tv_label = np.concatenate([train_label ,val_label])

vt_data = np.concatenate([val_data ,test_data])
vt_label = np.concatenate([val_label ,test_label])


# 数据预处理
np.save("train_data.npy" ,train_data)
np.save("val_data.npy" ,val_data)
np.save("test_data.npy" ,test_data)

np.save("train_label.npy" ,train_label)
np.save("val_label.npy" ,val_label)
np.save("test_label.npy" ,test_label)

np.save("tt_data.npy" ,tt_data)
np.save("tt_label.npy" ,tt_label)
np.save("tv_data.npy" ,tv_data)
np.save("tv_label.npy" ,tv_label)
np.save("vt_data.npy" ,vt_data)
np.save("vt_label.npy" ,vt_label)
