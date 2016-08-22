# encoding: utf-8
import defaultdict

N = 17997
lr_init = 0.1

def update(W, X, l, lr):
    a = sum([W[x] for x in X])
    g = ((1. / (1. + math.exp(-a))) - l) if -100. < a else (0. -l)
    for x in X:
        W[x] = W[x] - lr*g


def train(fi):
    t = 1
    W = collections.defaultdict(float)
    for line in fi:
        fields = line.strip('\n').split('\t')
        update(W, fields[1:], float(fields[0]), lr_init / (1+t/float(N)))
        t += 1
    return W
