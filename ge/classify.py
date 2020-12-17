from __future__ import print_function


import numpy
from sklearn.metrics import f1_score, accuracy_score,precision_score,classification_report,recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y,Y_)
        print('-------------------')
        print('准确率：',accuracy_score(Y,Y_))
        print('宏平均精确率:', precision_score(Y,Y_, average='macro'))  # 预测宏平均精确率输出
        print('微平均精确率:', precision_score(Y,Y_, average='micro'))  # 预测微平均精确率输出
        print('加权平均精确率:', precision_score(Y,Y_, average='weighted'))  # 预测加权平均精确率输出
        print('宏平均召回率:', recall_score(Y,Y_, average='macro'))  # 预测宏平均召回率输出
        print('微平均召回率:', recall_score(Y,Y_, average='micro'))  # 预测微平均召回率输出
        print('加权平均召回率:', recall_score(Y,Y_, average='micro'))  # 预测加权平均召回率输出

        print('宏平均F1-score:',
              f1_score(Y,Y_, labels=[0,1], average='macro'))  # 预测宏平均f1-score输出
        print('微平均F1-score:',
              f1_score(Y,Y_, labels=[0,1], average='micro'))  # 预测微平均f1-score输出
        print('加权平均F1-score:',
              f1_score(Y,Y_, labels=[0,1], average='weighted'))  # 预测加权平均f1-score输
        print('分类报告\n',classification_report(Y,Y_))
        print(results)
        return results
        print('-------------------')

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def read_node_label1(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(',')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y