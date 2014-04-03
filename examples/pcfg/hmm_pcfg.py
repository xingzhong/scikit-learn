print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pcfg import PCFG, Production, Terminal, Grammar, Nonterminal
from sklearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model

TEST_N = 30

def null_hmm():
    startprob = np.array([1.0])
    transmat = np.array([[1.0]])
    means = np.array([[0.0]])
    covars = np.tile(np.identity(1), (1, 1, 1))
    model = GaussianHMM(1, "full", startprob, transmat)
    model.means_ = means
    model.covars_ = covars
    return model

def postive_hmm():
    startprob = np.array([0.0, 1.0])
    transmat = np.array([[0.8, 0.2], [0.2, 0.8]])
    means = np.array([[1.0], [-1.0]])
    covars = np.tile(np.identity(1), (2, 1, 1))
    pModel = GaussianHMM(2, "full", startprob, transmat, init_params='st')
    pModel.means_ = means
    pModel.covars_ = covars
    return pModel

def null_pcfg(mean=0):
	S = Nonterminal('S')
	NT1 = Nonterminal('NT1')
	null = Terminal('null', np.array([[mean]]), np.array([[1]]))
	prods = [ ]
	prods.append(Production(S, [null, S], prob=.9))
	prods.append(Production(S, [null, null], prob=.1))
	return Grammar(S, prods)

def positive_pcfg(mean1=1, mean2=-1):
    S = Nonterminal('S')
    NT1 = Nonterminal('NT_left')
    NT2 = Nonterminal('NT_llow')
    NT3 = Nonterminal('NT_right')
    NT4 = Nonterminal('NT_high')
    
    high = Terminal('high', np.array([[mean1]]), np.array([[1]]))
    low = Terminal('low', np.array([[mean2]]), np.array([[1]]))
    prods = [ ]
    prods.append(Production(S, [NT1, S], prob=1))
    prods.append(Production(S, [NT1, NT1], prob=1e-3))
    prods.append(Production(NT1, [NT3, NT2], prob=1.0))
    prods.append(Production(NT3, [NT2, NT4], prob=1.0))
    #prods.append(Production(NT1, [NT2, NT4], prob=1.0))
    prods.append(Production(NT2, [low, NT2], prob=0.8))
    prods.append(Production(NT2, [low, low], prob=.2))
    prods.append(Production(NT4, [high, NT4], prob=.8))
    prods.append(Production(NT4, [high, high], prob=.2))
    #prods.append(Production(NT3, [NT3, NT3], prob=0.5))
    #prods.append(Production(NT3, [low, low], prob=0.5))
    return Grammar(S, prods)

def pretty(tree):
    if isinstance(tree, tuple):
        return "["+" ".join(map(pretty, tree))+"]"
    else :
        return str(tree)

def flatten(tree):
    leaf = []
    _flatten(tree, leaf)
    return leaf

def _flatten(tree, leaf):
    if len(tree) == 2:
        leaf.append(tree[0].means)
    else:
        map(lambda x: _flatten(x, leaf), tree[1:])
def maskIter(mask):
    i,j,k = mask.shape
    for ii in range(i):
        for jj in range(j):
            for kk in range(k):
                if mask[ii,jj,kk] != 0 :
                    yield (ii,jj,kk)

def evaluation(chuck, clf, scaler, model1, model2, model3, model4, signal):
    
    T = np.atleast_2d(range(N)).T
    clf.fit(T, chuck)
    meanless = np.atleast_2d(chuck - clf.predict(T)).T
    #import pdb; pdb.set_trace()
    sample = scaler.fit_transform(meanless)
    low = np.take(sample, np.where(signal[:N]==False))
    high = np.take(sample, np.where(signal[:N]))
    model3.means_[0] = np.mean(high)
    model3.means_[1] = np.mean(low)
    model4.means_[0] = np.mean(low)
    model1 = PCFG(positive_pcfg(mean1= np.mean(high), mean2=np.mean(low)))
    model2 = PCFG(null_pcfg(mean=np.mean(low)))
    logLik1, path1 = model1.decode(sample)
    logLik2, path2 = model2.decode(sample)
    logLik3, path3 = model3.decode(sample)
    logLik4, path4 = model4.decode(sample)

    #import pdb; pdb.set_trace()

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(311)
    ax2 = ax.twinx()
    ax.plot(range(N), chuck)
    ax.plot(range(N), clf.predict(T), '--r')
    ax2.step(range(N), signal[:N], color="k", ls="--")
    ax.grid()

    bx = fig.add_subplot(312)
    bx.plot(range(N), sample, label='obs')
    bx.step(range(N), np.array(map(lambda x: model3.means_[x], path3)), label='hmm')
    #bx.set_ylim([-0.05, 1.05])
    bx.grid()
    bx.set_title("[P]%.4f / [N]%.4f"%(logLik3, logLik4))

    cx = fig.add_subplot(313)
    cx.step(range(N), np.array(flatten(path1)).ravel(), label='pcfg')
    cx.plot(range(N), sample, label='obs')
    #cx.set_ylim([-0.05, 1.05])
    cx.grid()
    cx.set_title("[P]%.4f / [N]%.4f"%(logLik1, logLik2))
    fig.tight_layout()
    plt.show()
if __name__ == '__main__':
    import nibabel
    import pandas as pd
    N = 64
    func = "/home/xingzhong/nilearn_data/haxby2001/subj1/bold.nii.gz"
    mask = "/home/xingzhong/nilearn_data/haxby2001/subj1/mask4_vt.nii.gz"
    label = "/home/xingzhong/nilearn_data/haxby2001/subj1/labels.txt"
    labels = pd.read_csv(label, sep=' ')
    signal = labels.labels != 'rest'

    fmri_img = nibabel.load(func)
    fmri_data = fmri_img.get_data()
    mask_vt = nibabel.load(mask).get_data()
    #sample = np.atleast_2d([-1,-1,-1,1,1,1,-1,-1.0]*4).T + 0.6*np.random.randn(N,1)
    clf = linear_model.LinearRegression()
    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(-1, 2))
    model1 = PCFG(positive_pcfg())
    model2 = PCFG(null_pcfg())
    model3 = postive_hmm()
    model4 = null_hmm()
    for (i,j,k) in maskIter(mask_vt):
        print i,j,k
        chuck = fmri_data[i,j,k,:N].astype(float)
        evaluation(chuck, clf, scaler, model1, model2, model3, model4, signal)