import Bio.SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis                  # 用于计算蛋白质的一些特性
from propy import PseudoAAC                                         # 用于计算蛋白质的一些特性
import numpy as np
from multiprocessing import Pool ,cpu_count
import networkx as nx
import math
from Bio import SeqIO
from propy import CTD


# 加载数据
train_data = np.load("train_data.npy")
val_data = np.load("val_data.npy")
test_data = np.load("test_data.npy")
train_label = np.load("train_label.npy")
val_label = np.load("val_label.npy")
test_label = np.load("test_label.npy")
tt_data = np.load("tt_data.npy")
tt_label = np.load("tt_label.npy")
tv_data = np.load("tv_data.npy")
tv_label = np.load("tv_label.npy")
vt_data = np.load("vt_data.npy")
vt_label = np.load("vt_label.npy")

print("train shape" ,train_data.shape)
print("val shape" ,val_data.shape)
print("test shape" ,test_data.shape)

print("train model")

from catboost import CatBoostClassifier

clf = CatBoostClassifier(loss_function="Logloss",
                           # eval_metric="AUC",
                           eval_metric="Accuracy",
                           task_type="GPU",
                           devices="0",
                           # random_seed=432013,
                           learning_rate=0.01,
                           iterations=70000,
                           l2_leaf_reg=49,
                           od_type="Iter",
                           depth=9,
                           early_stopping_rounds=15000,
                           border_count=64)

clf.fit(train_data,
        train_label,
        # eval_set=(val_data,val_label),
        # eval_set=(test_data,test_label)
        eval_set=(vt_data,vt_label))

pred_res = clf.predict_proba(test_data)[:,1]

label = []
for x in pred_res:
    if x > 0.5:
        label.append(1)
    else:
        label.append(0)

from sklearn import metrics

tn ,fp ,fn ,tp = metrics.confusion_matrix(test_label ,label).ravel()
precise = metrics.precision_score(test_label ,label)
acc = metrics.accuracy_score(test_label ,label)
f1 = metrics.f1_score(test_label ,label)
recall = metrics.recall_score(test_label ,label)
mcc = metrics.matthews_corrcoef(test_label ,label)
auc = metrics.roc_auc_score(test_label ,pred_res)
ap = metrics.average_precision_score(test_label ,pred_res)

sn = tp / (tp + fn)
sp = tn / (fp + tn)

print("tn:" ,tn)
print("tp:" ,tp)
print("fp:" ,fp)
print("fn:" ,fn)
print("sn:" ,sn)
print("sp:" ,sp)
print("ACC:" ,acc)
print("precise:" ,precise)
print("f1:" ,f1)
print("recall:" ,recall)
print("mcc:" ,mcc)
print("auc:" ,auc)
print("ap:" ,ap)
