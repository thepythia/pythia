#############SVM######################
1. preprocessing by fp2vector.py

trainvec.csv被读到Dictionary list,特征的排序已被打乱.
取出id, target, weights (var11),将剩余的列作为特征传给svm进行训练.
对于特征值，做如下操作：
1) 把nomial值转换成one-hot encoding, 即001,010,100,if only exists 3 distinct value
2) 缺失值用对应列的mean替换
3) (未做) 标准化已做过，故我们这边无需再做

相同的操作，在testvec.csv上做一遍.

2. 将转换好的数据传给SVM进行训练

1) id, target, weight单独抽取，不做为特征
2) 当前随便在用rbf, c=1.0, gamma=0.1, cache_size=2000
3) gridSearchCV想做，但由于10万的数据，上述配置的SVM当前训练出模型都很困难，所以...
4) 跑出一个模型后，再做Normalized Weighted Gini的评测，用训练集的一部分做为测试数据来测

###############Decision Tree###############
1. preprocessing by



