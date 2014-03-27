"""
==================================
Demonstration of parsing from PCFG
==================================

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pcfg import PCFG, Production, Terminal, Grammar, Nonterminal
from nltk.draw.tree import draw_trees


TEST_N = 30


def sample_grammar3():
    S = Nonterminal('S')
    NT1 = Nonterminal('NT_left')
    NT2 = Nonterminal('NT_llow')
    NT3 = Nonterminal('NT_right')
    NT4 = Nonterminal('NT_high')
    
    high = Terminal(np.array([[1]]), np.array([[.1]]))
    low = Terminal(np.array([[0]]), np.array([[.1]]))
    prods = [ ]
    prods.append(Production(S, [NT1, S], prob=.3))
    prods.append(Production(S, [NT1, NT1], prob=.7))
    prods.append(Production(NT1, [NT2, NT4, NT2], prob=1.0))
    #prods.append(Production(NT1, [NT2, NT4], prob=1.0))
    prods.append(Production(NT2, [low, NT2], prob=0.9))
    prods.append(Production(NT2, [low], prob=0.1))
    prods.append(Production(NT4, [high, NT4], prob=0.9))
    prods.append(Production(NT4, [high], prob=0.1))
    #prods.append(Production(NT3, [NT3, NT3], prob=0.5))
    #prods.append(Production(NT3, [low, low], prob=0.5))
    

    return Grammar(S, prods)

def sample_grammar2():
    S = Nonterminal('S')
    NT1 = Nonterminal('NT1')
    NT2 = Nonterminal('NT2')
    NT3 = Nonterminal('NT3')
    NT4 = Nonterminal('NT4')
    
    mid = Terminal(np.array([[0.5, 0.5]]), np.array([[1.0, 1.0]]))
    high = Terminal(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]))
    low = Terminal(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
    prods = [ ]
    prods.append(Production(S, [NT1, NT2], prob=1.0))
    prods.append(Production(NT1, [NT2, NT4], prob=1.0))
    prods.append(Production(NT2, [low, NT2], prob=0.8))
    prods.append(Production(NT2, [low, low], prob=0.2))
    prods.append(Production(NT4, [high, NT4], prob=0.8))
    prods.append(Production(NT4, [high, high], prob=0.2))

    return Grammar(S, prods)

def sample_grammar():
    S = Nonterminal('S')
    NT1 = Nonterminal('NT1')
    NT2 = Nonterminal('NT2')
    NT3 = Nonterminal('NT3')
    NT4 = Nonterminal('NT4')
    NT5 = Nonterminal('NT5')
    t1 = Terminal(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]))
    t2 = Terminal(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
    t3 = Terminal(np.array([[-1.0, -1.0]]), np.array([[1.0, 1.0]]))
    prods = [ Production(S, [S, NT1], prob=0.8) ]
    prods.append(Production(S, [NT2, NT3], prob=0.2) )
    prods.append(Production(NT1, [NT2, NT3], prob=1.0))
    prods.append(Production(NT2, [NT4, NT5], prob=1.0))
    prods.append(Production(NT3, [t3, NT3], prob=0.8))
    prods.append(Production(NT3, [t3, t3], prob=0.2))
    prods.append(Production(NT4, [t1, NT4], prob=0.8))
    prods.append(Production(NT4, [t1, t1], prob=0.2))
    prods.append(Production(NT5, [t2, NT5], prob=0.8))
    prods.append(Production(NT5, [t2, t2], prob=0.2))
    
    return Grammar(S, prods)

def sample_series():
    from pandas.io.data import DataReader
    import datetime
    start = datetime.datetime(2013, 2, 10)
    end = datetime.datetime(2014, 2, 20)
    #df = DataReader('vxx', 'yahoo', start, end)
    #X = np.atleast_2d(df[['Adj Close', 'Volume']].values)
    #import pdb; pdb.set_trace()
    #return X[:TEST_N, :]
    sample = np.atleast_2d([0,0,0,1,1,1,1,0,0, 0.0]*3).T
    #import pdb; pdb.set_trace()
    return sample

if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler
    grammar = sample_grammar3()
    model = PCFG(grammar)
    X = sample_series()
    scaler = MinMaxScaler(feature_range=(0, 1))
    normX = scaler.fit_transform(X)
    parses, expectations = model.parse(normX, n=1, trace=0, trim=10)
    #import pdb;pdb.set_trace()
    print '  please wait...'
    #draw_trees(*parses)
    #import pdb; pdb.set_trace()
    for s in parses[0].subtrees(lambda t: t.height()==2):
        print s.node, s.leaves()
    decode = [s.node.means for s in parses[0].subtrees(lambda t: t.height()==2)]
    decode = np.array(decode).ravel()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(TEST_N), normX, '-o')
    ax.bar(range(TEST_N), decode, alpha=0.4)
    ax.grid()
    plt.show()
