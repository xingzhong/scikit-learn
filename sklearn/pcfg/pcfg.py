import numpy as np
from sklearn.base import BaseEstimator
from nltk.grammar import WeightedProduction, WeightedGrammar, Nonterminal
from sklearn.mixture import log_multivariate_normal_density, sample_gaussian
from sklearn.utils import check_random_state
from scipy.misc import logsumexp
from hashlib import sha1
from nltk.parse.earleychart import ( IncrementalChartParser,
                                    CompleterRule,
                                    ScannerRule,
                                    PredictorRule)
from nltk.parse.chart import TopDownPredictRule, Chart, AbstractChartRule, LeafEdge, TreeEdge
from nltk.parse.pchart import ( ProbabilisticTreeEdge,
                                BottomUpProbabilisticChartParser)                              


GAMMA = Nonterminal('GAMMA')

class ProbabilisticLeafEdge(LeafEdge):
    def prob(self): return 1.0

class ProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES=0
    def apply_iter(self, chart, grammar, left_edge, right_edge):
        # Make sure the rule is applicable.
        if not (left_edge.end() == right_edge.start() and
                left_edge.next() == right_edge.lhs() and
                left_edge.is_incomplete() and right_edge.is_complete()):
            return

        # Construct the new edge.
        p = left_edge.prob() * right_edge.prob()
        new_edge = ProbabilisticTreeEdge(p,
                            span=(left_edge.start(), right_edge.end()),
                            lhs=left_edge.lhs(), rhs=left_edge.rhs(),
                            dot=left_edge.dot()+1)

        # Add it to the chart, with appropriate child pointers.
        changed_chart = False
        for cpl1 in chart.child_pointer_lists(left_edge):
            if chart.insert(new_edge, cpl1+(right_edge,)):
                changed_chart = True

        # If we changed the chart, then generate the edge.
        if changed_chart: yield new_edge

class ScannerRule(AbstractChartRule):
    NUM_EDGES=1
    def apply_iter(self, chart, grammar, edge, index=-1):
        if isinstance(edge.next(), Terminal):
            import pdb; pdb.set_trace()
            logProb = edge.next().logP(np.atleast_2d( chart.leaf(index)) )[0][0]
            new_edge = ProbabilisticTreeEdge.from_production(
                Production(edge.next(), [index], prob=1.0), 
                edge.end(), logProb)
            if chart.insert(new_edge, ()):
                yield new_edge

class PredictorRule(TopDownPredictRule):
    NUM_EDGES = 1
    def apply_iter(self, chart, grammar, edge):
        if edge.is_complete(): return
        for prod in grammar.productions(lhs=edge.next()):
            new_edge = ProbabilisticTreeEdge.from_production(prod, edge.end(), prod.prob())
            if chart.insert(new_edge, ()):
                yield new_edge

class Parser(BottomUpProbabilisticChartParser):
    def nbest_parse(self, tokens, n=None):
        self._grammar.check_coverage(tokens)
        chart = Chart(list(tokens))
        grammar = self._grammar
        
        # Chart parser rules.
        pr = PredictorRule()
        sc = ScannerRule()

        # Our queue!
        queue = []

        nullEdge = ProbabilisticTreeEdge.from_production(
                            Production(GAMMA, [S], prob=1.0),
                            0, 1.0)
        queue.append(nullEdge)
        chart.insert(nullEdge, ())
        for (i, token) in enumerate(tokens):
            for edge in chart.iteredges():
                queue.extend(pr.apply(chart, grammar, edge))
            for edge in chart.iteredges():
                queue.extend(sc.apply(chart, grammar, edge, i))
            
            if self._trace > 0:
                for edge in chart.iteredges():
                    print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),
                                        edge.prob()))
            import pdb; pdb.set_trace()
            

        # Get a list of complete parses.
        parses = chart.parses(grammar.start(), ProbabilisticTree)

        # Assign probabilities to the trees.
        prod_probs = {}
        for prod in grammar.productions():
            prod_probs[prod.lhs(), prod.rhs()] = prod.prob()
        for parse in parses:
            self._setprob(parse, prod_probs)

        # Sort by probability
        parses.sort(reverse=True, key=lambda tree: tree.prob())

        return parses[:n]

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
    def check_coverage(self, tokens):
        _, m = self._lexical_index.keys()[0].means.shape
        _, n = tokens.shape
        if m != n :
            raise ValueError("Grammar does not cover the inputs dim")


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

    

    def parse(self, X):
        p = Parser(self.grammar, trace=1)
        p.nbest_parse(X)

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
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0] ])
    model.parse(X)