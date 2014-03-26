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
from nltk.tree import Tree, ProbabilisticTree
from collections import Counter

GAMMA = Nonterminal('GAMMA')

class Sample(int):
    def __repr__(self):
        return "S[%s]"%str(self)

class ChartI(Chart):
    def delete(self, edge):
        self._edges.remove(edge)
        #self._edge_to_cpls.pop(edge, None)
        for subdict in self._indexes.values():
            for edges in subdict.values():
                try:
                    edges.remove(edge)
                except ValueError:
                    pass
        
    def _trees(self, edge, complete, memo, tree_class):
        # If we've seen this edge before, then reuse our old answer.
        if edge in memo:
            return memo[edge]

        trees = []

        # when we're reading trees off the chart, don't use incomplete edges
        if complete and edge.is_incomplete():
            return trees

        # Until we're done computing the trees for edge, set
        # memo[edge] to be empty.  This has the effect of filtering
        # out any cyclic trees (i.e., trees that contain themselves as
        # descendants), because if we reach this edge via a cycle,
        # then it will appear that the edge doesn't generate any
        # trees.
        memo[edge] = []

        # Leaf edges.

        if isinstance(edge.lhs(), Terminal):
            #import pdb; pdb.set_trace()
            #leaf = self._tokens[edge.start()]
            leaf = np.array_str(self._tokens[edge.start()], precision=3, max_line_width=6)
            t = tree_class(edge.lhs(), [leaf])
            t.set_logProb(edge.logProb())
            memo[edge] = [t]
            return [t]

        # Each child pointer list can be used to form trees.
        for cpl in self.child_pointer_lists(edge):
            # Get the set of child choices for each child pointer.
            # child_choices[i] is the set of choices for the tree's
            # ith child.
            child_choices = [self._trees(cp, complete, memo, tree_class)
                             for cp in cpl]

            # For each combination of children, add a tree.
            for children in self._choose_children(child_choices):
                lhs = edge.lhs().symbol()
                trees.append(tree_class(lhs, children))

        # If the edge is incomplete, then extend it with "partial trees":
        if edge.is_incomplete():
            unexpanded = [tree_class(elt,[])
                          for elt in edge.rhs()[edge.dot():]]
            for tree in trees:
                tree.extend(unexpanded)

        # Update the memoization dictionary.
        memo[edge] = trees

        # Return the list of trees.
        return trees

class ProbabilisticTreeI(ProbabilisticTree):
    def __init__(self, *args, **kwargs):
        ProbabilisticTree.__init__(self, *args, **kwargs)
        self._logProb = None
        
    def logProb(self):
        return self._logProb
    def set_logProb(self, x):
        self._logProb = x
        self.set_prob(np.exp(x))

class ProbabilisticTreeEdgeI(TreeEdge):
    def __init__(self, logProb, *args, **kwargs):
        TreeEdge.__init__(self, *args, **kwargs)
        self._logProb = logProb
        self._alpha = 0  # forward 
        self._gamma = 0   # inner
    
    def __cmp__(self, other):
        if self._logProb != other.logProb(): return -1
        return TreeEdge.__cmp__(self, other)

    def move_dot_forward(self, new_end):
        return ProbabilisticTreeEdgeI(self._logProb, 
                        span=(self._span[0], new_end),
                        lhs=self._lhs, rhs=self._rhs,
                        dot=self._dot+1)

    def prob(self): return np.exp(self._logProb)
    def logProb(self): return self._logProb

    @staticmethod
    def from_production(production, index, logP):
        return ProbabilisticTreeEdgeI(logP, (index, index), production.lhs(),
                                     production.rhs(), 0)

class ProbabilisticLeafEdgeI(LeafEdge):
    def prob(self): return 1.0
    def logProb(self): return 0.0

class ProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES=0
    def apply_iter(self, chart, grammar, left_edge, right_edge):
        #import pdb; pdb.set_trace()
        # Make sure the rule is applicable.
        if not (left_edge.end() == right_edge.start() and
                left_edge.next() == right_edge.lhs() and
                left_edge.is_incomplete() and right_edge.is_complete()):
            return

        # Construct the new edge.
        logP = left_edge.logProb() + right_edge.logProb()

        new_edge = ProbabilisticTreeEdgeI(logP,
                            span=(left_edge.start(), right_edge.end()),
                            lhs=left_edge.lhs(), rhs=left_edge.rhs(),
                            dot=left_edge.dot()+1)
        new_edge._alpha = left_edge._alpha + right_edge._gamma
        new_edge._gamma = left_edge._gamma + right_edge._gamma
        # Add it to the chart, with appropriate child pointers.
        changed_chart = False
        for cpl1 in chart.child_pointer_lists(left_edge):
            if chart.insert(new_edge, cpl1+(right_edge,)):
                changed_chart = True

        # If we changed the chart, then generate the edge.
        if changed_chart: yield new_edge

class SingleCompleteRule(AbstractChartRule):
    NUM_EDGES=1
    _fundamental_rule = ProbabilisticFundamentalRule()
    def apply_iter(self, chart, grammar, edge1, trim):
        fr = self._fundamental_rule
        if edge1.is_complete():
            edges = [edge for edge in chart.select(end=edge1.start(), 
                                        is_complete=False,
                                        next=edge1.lhs())]

            for edge2 in sorted(edges, key=lambda x:x._alpha, reverse=True)[:trim]:
                # FIXME: This loop is too heavy  
                for new_edge in fr.apply_iter(chart, grammar, edge2, edge1):
                    yield (new_edge, edge2, edge1)


    def __str__(self): return 'Fundamental Rule'

class ScannerRule(AbstractChartRule):
    NUM_EDGES=1
    def apply_iter(self, chart, grammar, edge, index=-1):
        if isinstance(edge.next(), Terminal):
            logProb = edge.next().logP(np.atleast_2d( chart.leaf(index)) )[0][0]
            new_edge = ProbabilisticTreeEdgeI.from_production(
                Production(edge.next(), [Sample(index)], prob=1.0), 
                index, logProb)
            new_edge = new_edge.move_dot_forward(index+1)
            new_edge._alpha = edge._alpha + logProb
            new_edge._gamma = edge._gamma + logProb
            if chart.insert(new_edge, ()):
                #print('[posterior]  %-50s [%.4f]' % (chart.pp_edge(new_edge,width=2),
                #                    new_edge.logProb()))
                yield new_edge

class PredictorRule(TopDownPredictRule):
    NUM_EDGES = 1
    def apply_iter(self, chart, grammar, edge):
        if edge.is_complete(): return
        for prod in grammar.productions(lhs=edge.next()):
            new_edge = ProbabilisticTreeEdgeI.from_production(
                prod, edge.end(), prod.logProb())
            new_edge._alpha = edge._alpha + new_edge.logProb()
            new_edge._gamma = new_edge.logProb()
            #print('[prior    ]  %-50s [%.4f] [%.4f] [%.4f]' % (chart.pp_edge(new_edge,width=2),
            #                        new_edge.logProb(), new_edge._alpha, new_edge._gamma))
            #print('[priorfrom]  %-50s [%.4f] [%.4f] [%.4f]' % (chart.pp_edge(edge,width=2),
            #                        edge.logProb(), edge._alpha, edge._gamma))
            if chart.insert(new_edge, ()):
                yield new_edge

class Parser(BottomUpProbabilisticChartParser):
    def _setLogProb(self, tree, prod_probs):
        if tree.logProb() is not None: return 
        
        # Get the prob of the CFG production.
        lhs = Nonterminal(tree.node)
        rhs = []
        for child in tree:
            if isinstance(child, Tree):
                rhs.append(Nonterminal(child.node))
            else:
                rhs.append(child)
        
        logProb = np.log(prod_probs[lhs, tuple(rhs)])

        # Get the probs of children.
        for child in tree:
            if isinstance(child, Tree):
                self._setLogProb(child, prod_probs)
                logProb += child.logProb()

        tree.set_logProb(logProb)

    def nbest_parse(self, tokens, n=None, trim=10):
        self._grammar.check_coverage(tokens)
        chart = ChartI(list(tokens))
        grammar = self._grammar
        
        # Chart parser rules.
        pr = PredictorRule()
        sc = ScannerRule()
        cl = SingleCompleteRule()

        # Our queue!
        #queue = []

        nullEdge = ProbabilisticTreeEdgeI.from_production(
                            Production(GAMMA, [self._grammar.start()], prob=1.0),
                            0, 0.0)
        #queue.append(nullEdge)
        chart.insert(nullEdge, ())
        expectations = []
        for (i, token) in enumerate(tokens):

            for edge in chart.select(end=i, is_incomplete=True):
                pr.apply(chart, grammar, edge)

            predict = Counter()
            for edge in chart.select(end=i, is_incomplete=True):
                if isinstance(edge.next(), Terminal):
                    predict[edge.next()] += np.exp(edge._alpha)

            normalize = sum(predict.values())
            expectation = 0
            #print 
            for k,v in predict.iteritems():
                #print k, v, v/normalize
                expectation += v/normalize * k.means
            expectations.append(expectation.ravel())

            for edge in chart.select(end=i, is_incomplete=True):
                sc.apply(chart, grammar, edge, i)
            #for edge in chart.select(end=i+1):

            for edge in chart.select(end=i+1, is_complete=True):
                cl.apply(chart, grammar, edge, trim)


            #for nt in grammar._categories.union([GAMMA]):
            #    edges = [e for e in chart.select(end=i+1, is_complete=True, lhs=nt)]
            #    if len(edges) > 10:
            #        edges = sorted(edges, key=lambda x: x.logProb())
            #        for edge in edges[:-10]:
            #            print('[delete]  %-50s [%s]' % (chart.pp_edge(edge,width=2),
            #                            edge.logProb()))
            #            chart.delete(edge)


            
            if self._trace > 1:
                print "="*25+"Prior     "+str(i)+"="*25
                for edge in chart.select(end=i, is_incomplete=True):
                    print('%-50s [%.4f] [%.4f] [%.4f]' % (chart.pp_edge(edge,width=2),
                                    edge.logProb(), edge._alpha, edge._gamma))
                print "="*25+"Posterior "+str(i)+"="*25
                #for edge in chart.select(end=i+1, is_complete=True):
                for edge in chart.select(end=i+1):
                    print('%-50s [%.4f] [%.4f] [%.4f]' % (chart.pp_edge(edge,width=2),
                                    edge.logProb(), edge._alpha, edge._gamma))
                print "="*25+"char edges "+str(chart.num_edges())+"="*25
            
            #import pdb;pdb.set_trace()
        # Get a list of complete parses.
        parses = chart.parses(self._grammar.start(), ProbabilisticTreeI)
        
        for edge in chart.select(end=i+1, lhs=GAMMA):
            chart._edges.remove(edge)
            #print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),
            #                            edge.logProb()))
        #import pdb;pdb.set_trace()
        # Assign probabilities to the trees.
        prod_probs = {}
        for prod in grammar.productions():
            prod_probs[prod.lhs(), prod.rhs()] = prod.prob()
        
        for parse in parses:
            self._setLogProb(parse, prod_probs)
        #import pdb; pdb.set_trace()
        # Sort by probability

        print "total parse trees", len(parses)

        parses.sort(reverse=True, key=lambda tree: tree.logProb())
        
        return parses[:n], expectations

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

    def symbol(self):
        return self

    def __str__(self):
        return "Gau[%s]"%np.array_str(self.means.ravel(), precision=3)

    def __repr__(self):
        return "Gau[%s]"%np.array_str(self.means.ravel(), precision=3)

    def __hash__(self):
        means = sha1(self.means.view(np.uint8)).hexdigest()
        covars = sha1(self.covars.view(np.uint8)).hexdigest()
        return hash(means + covars)

    def __eq__(self, other):
        return hash(self) == hash(other)


class Production(WeightedProduction):
    def logProb(self):
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
                prob = prod.logProb()
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

    def parse(self, X, n=None, trim=10, trace=None):
        p = Parser(self.grammar, trace=trace)
        parses, expectations = p.nbest_parse(X, n=n, trim=trim)
        return parses, np.array(expectations)
        

if __name__ == '__main__':
    np.set_printoptions(precision=4)
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
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0] ])
    X = np.random.randn(8,2)
    parses = model.parse(X)
    #import pdb; pdb.set_trace()
    #from nltk.draw.tree import draw_trees
    #print '  please wait...'
    #draw_trees(*parses)