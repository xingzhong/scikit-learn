import numpy as np
from sklearn.base import BaseEstimator
from nltk.grammar import WeightedProduction, WeightedGrammar, Nonterminal
from sklearn.mixture import log_multivariate_normal_density, sample_gaussian
from sklearn.utils import check_random_state
from scipy.misc import logsumexp
from hashlib import sha1


class Terminal(object):
    """ Terminal
    A terminal is a multivariate normal distribution determined by mean and covars
    """
    def __init__(self, means, covars):
        self.means = means 
        self.covars = covars

    def sample(self):
        return sample_gaussian(self.means.ravel(), self.covars.ravel())

    def logP(self, x):
        return log_multivariate_normal_density(x, self.means, self.covars)

    def __str__(self):
        return "Terminal"

    def __repr__(self):
        return "Terminal"

    def __hash__(self):
        means = sha1(self.means.view(np.uint8)).hexdigest()
        covars = sha1(self.covars.view(np.uint8)).hexdigest()
        return hash(means + covars)

    def __eq__(self, other):
        return hash(self) == hash(other)


class Production(WeightedProduction):
    def logprob(self):
        # use natural log instead log2 in nltk
        return np.log(self.prob())

class Grammar(WeightedGrammar):
    pass


class PCFG(BaseEstimator):
    """Probabilistic (Stochastic) Context-free grammar 

    This class evaluate the PCFG on real data.

    """
    def __init__(self, grammar, random_state=None):
        self.grammar = grammar
        self.inside_ = {}
        self.random_state = random_state

    def score(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        """
        self._X = X
        self.N, _ = X.shape
        self._inside()
        start = self.grammar.start()
        self.inside_ = {}

        return self.inside(0, self.N-1, start)

    def _inside(self):
        terminals = self.grammar._lexical_index.keys()
        self.B = { terminal: terminal.logP(self._X) for terminal in terminals}
        

    def inside(self, s, t, i):
        """Calculate the inside probability

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        s: int 
            Index of inside probability
        t: int 
            Index of inside probability
        i: Nonterminal
            Nonterminal who generate X[s] to X[t]
           or Terminal who generate X[s=t]
        """
        if s > t : raise IndexError("s should smaller than t")
        if self.inside_.has_key( (s, t, i) ):
            return self.inside_[(s, t, i)]
        if s == t:
            self.inside_[(s, t, i)] = self.B.get(i, -np.inf*np.ones((self.N, 1)))[s][0]
            return self.inside_[(s, t, i)]
        else:
            logs = []
            for prod in self.grammar.productions(lhs=i):
                j, k = prod.rhs()
                prob = prod.logprob()
                for r in range(s, t):
                    logProb = prob + self.inside(s, r, j) + self.inside(r+1, t, k)                
                    if np.isfinite( logProb ):
                        logs.append( logProb )
            if len(logs) > 0:
                self.inside_[(s, t, i)] = logsumexp(np.array(logs))
            else :
                self.inside_[(s, t, i)] = -np.inf
                
            return self.inside_[(s, t, i)]
    
    
    def _sample(self, root):
        
        if isinstance(root, Terminal):
            return np.atleast_2d(root.sample())

        temp = [ (prod, prod.prob() ) for prod in self.grammar.productions(lhs=root)]
        prods, probs = zip(*temp)
        prob = np.random.choice( prods, 1, p=probs)[0]
        return np.vstack ( map(lambda x: self._sample(x), prob.rhs()) )

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)
        

        return self._sample(self.grammar.start())[:n_samples, :]

    def prior(self, X):
        """Predict the prior probability 
            of the next nonterminal and observation based on current observations
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of observation samples
        """
        pass



if __name__ == '__main__':
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
    
    grammar = Grammar(S, prods)
    model = PCFG(grammar)
    
    import pdb; pdb.set_trace()