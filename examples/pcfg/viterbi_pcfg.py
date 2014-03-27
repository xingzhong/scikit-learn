print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pcfg import PCFG, Production, Terminal, Grammar, Nonterminal

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

if __name__ == '__main__':
	#sample = np.atleast_2d([0,0,0,1,1,1,1,0,0, 0.0]*2).T
	sample = np.random.randn(60,1)
	model = PCFG(sample_grammar3())
	print model.viterbi(sample)
	#print model.score(sample)


