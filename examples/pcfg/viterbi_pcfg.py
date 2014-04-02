print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pcfg import PCFG, Production, Terminal, Grammar, Nonterminal

TEST_N = 30

def sample_grammar():
	S = Nonterminal('S')
	NT1 = Nonterminal('NT1')
	null = Terminal('null', np.array([[0]]), np.array([[.5]]))
	prods = [ ]
	prods.append(Production(S, [NT1, S], prob=.3))
	prods.append(Production(S, [null, null], prob=.7))
	prods.append(Production(NT1, [null, null], prob=1.0))
	return Grammar(S, prods)

def sample_grammar3():
    S = Nonterminal('S')
    NT1 = Nonterminal('NT_left')
    NT2 = Nonterminal('NT_llow')
    NT3 = Nonterminal('NT_right')
    NT4 = Nonterminal('NT_high')
    
    high = Terminal('high', np.array([[1]]), np.array([[.5]]))
    low = Terminal('low', np.array([[-1]]), np.array([[.5]]))
    prods = [ ]
    prods.append(Production(S, [NT1, S], prob=.3))
    prods.append(Production(S, [NT1, NT1], prob=.7))
    prods.append(Production(NT1, [NT3, NT2], prob=1.0))
    prods.append(Production(NT3, [NT2, NT4], prob=1.0))
    #prods.append(Production(NT1, [NT2, NT4], prob=1.0))
    prods.append(Production(NT2, [low, NT2], prob=0.9))
    prods.append(Production(NT2, [low, low], prob=0.1))
    prods.append(Production(NT4, [high, NT4], prob=0.9))
    prods.append(Production(NT4, [high, high], prob=0.1))
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

if __name__ == '__main__':
    sample = np.atleast_2d([-1,-1,-1,1,1,1,-1,-1.0]*4).T + 1.2*np.random.randn(32,1)
    #sample = np.random.randn(32,1)
    model1 = PCFG(sample_grammar3())
    model2 = PCFG(sample_grammar())
    lik1, t1 = model1.viterbi(sample)
    s1 = np.array(flatten(t1)).ravel()
    #import pdb; pdb.set_trace()
    lik2, t2 = model2.viterbi(sample)
    print "positive model", lik1
    print "negative model", lik2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(32), sample)
    ax.step(range(32), s1)
    plt.show()

