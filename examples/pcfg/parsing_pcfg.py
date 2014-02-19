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
    start = datetime.datetime(2013, 5, 10)
    end = datetime.datetime(2014, 1, 28)
    df = DataReader('cri', 'yahoo', start, end)
    X = np.atleast_2d(df[['Adj Close', 'Volume']].values)
    #import pdb; pdb.set_trace()
    return X[:20, :]
    return np.random.randn(15,2)

if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler
    grammar = sample_grammar()
    model = PCFG(grammar)
    X = sample_series()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normX = scaler.fit_transform(X)
    parses = model.parse(normX, n=1, trace=1)
    
    print '  please wait...'
    draw_trees(*parses)
    #import pdb; pdb.set_trace()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bx = ax.twinx()
    ax.plot(range(15), normX[:, 0], '-o')
    bx.bar(range(15), normX[:, 1], alpha=0.5)
    plt.show()
