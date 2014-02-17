import unittest
import numpy as np
from nose.tools import assert_equals, assert_true
from sklearn.pcfg import PCFG, Production, Terminal, Grammar, Nonterminal
from numpy.testing import (assert_array_almost_equal, assert_approx_equal,
                           assert_raises)

class TestPCFG(unittest.TestCase):
	def testDummy(self):
		grammar = 123
		model = PCFG(123)
		assert_equals( model.grammar, 123 )

	def testProduction(self):
		lhs = Nonterminal('a')
		t1 = Terminal(np.array([[1.0, 0.0]]), np.array([[1.0, 0.5]]))
		t2 = Terminal(np.random.rand(1,2), np.ones((1,2)))
		rhs = [t1, t2]
		prob = 0.8
		prod = Production(lhs, rhs, prob=prob)
		assert_equals( prod.lhs(), Nonterminal('a') )
		assert_equals( len(prod), 2)
		assert_true( prod.is_lexical() )

	def testTerminal_1(self):
		means = np.array([[0.0]])
		covars = np.array([[1.0]])
		t = Terminal(means, covars)
		assert_approx_equal( 
			t.logP(np.array([[0.0]]))[0][0], 
			np.log(0.398942),
			significant=5)

	def testTerminal_2(self):
		means = np.array([[1.0, 0.0]])
		covars = np.array([[1.0, 0.5]])
		t = Terminal(means, covars)
		assert_array_almost_equal( 
			t.logP(np.array([[0.0, 1.2], [2.1, -0.4]])), 
			np.array([[-3.43130348], [-2.25630348]]),
			decimal=4)

	def testTerminal_3(self):
		means = np.array([[1.0, 0.0]])
		covars = np.array([[1.0, 0.5]])
		t1 = Terminal(means, covars)
		means = np.array([[1.0, 0.0]])
		covars = np.array([[1.0, 0.5]])
		t2 = Terminal(means, covars)
		assert_true( t1 == t2, msg='{0}, {1}'.format(hash(t1), hash(t2)) )

	def testGrammar(self):
		start = Nonterminal('S')
		nt1 = Nonterminal('NT1')
		nt2 = Nonterminal('NT2')
		t1 = Terminal(np.random.rand(1,2), np.ones((1,2)))
		t2 = Terminal(np.random.rand(1,2), np.ones((1,2)))
		t3 = Terminal(np.random.rand(1,2), np.ones((1,2)))
		t4 = Terminal(np.random.rand(1,2), np.ones((1,2)))
		t5 = Terminal(np.random.rand(1,2), np.ones((1,2)))
		prod1 = Production(start, [t1, nt1], prob=0.8)
		prod2 = Production(start, [nt2, t4], prob=0.2)
		prod3 = Production(nt1, [nt2, t2, t3], prob=0.6)
		prod4 = Production(nt1, [t5], prob=0.4)
		prod5 = Production(nt2, [nt2, t4], prob=0.4)
		prod6 = Production(nt2, [t4, t3], prob=0.6)
		grammar = Grammar(start, [prod1, prod2, prod3, prod4, prod5, prod6])
		assert_equals( grammar.start(), start)
		assert_equals( len(grammar.productions()), 6)
		assert_equals( len(grammar.productions(lhs=start)), 2)
		assert_equals( len(grammar.productions(rhs=nt2)), 3)
		assert_equals( len(grammar.productions(lhs=nt2, rhs=nt2)), 1)
		assert_equals( len(grammar.productions(lhs=start, rhs=nt1)), 0)

	def testPCFG(self):
		start = Nonterminal('S')
		nt1 = Nonterminal('NT1')
		
		t1 = Terminal(np.random.rand(1,2), np.ones((1,2)))
		t2 = Terminal(np.random.rand(1,2), np.ones((1,2)))
		
		prod1 = Production(start, [nt1, t1], prob=0.8)
		prod2 = Production(start, [t2, t1], prob=0.2)
		prod3 = Production(nt1, [t1, t2], prob=1.0)
		
		grammar = Grammar(start, [prod1, prod2, prod3])
		assert_equals( grammar.start(), start)

		model = PCFG(grammar)
		assert_equals( model.grammar, grammar)

	
	def testInside(self):
		start = Nonterminal('S')
		t1 = Terminal(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
		prod1 = Production(start, [start, t1], prob=0.8)
		prod2 = Production(start, [t1, t1], prob=0.2)
		
		grammar = Grammar(start, [prod1, prod2])
		model = PCFG(grammar)
		X = np.array([[0.0, 0.0], [0.0, 0.0] ])
		score = model.score(X)
		assert_array_almost_equal( 
			model.inside(0, 0, t1), 
			np.array(-1.837877),
			decimal=4)
		assert_array_almost_equal( 
			score, 
			np.array(-5.2851919),
			decimal=4)
		X = np.random.randn(50,2)
		score = model.score(X)
		assert_true(np.isfinite(score))

	def testSample(self):
		start = Nonterminal('S')
		t1 = Terminal(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
		prod1 = Production(start, [start, t1], prob=0.8)
		prod2 = Production(start, [t1, t1], prob=0.2)
		
		grammar = Grammar(start, [prod1, prod2])
		model = PCFG(grammar)
		samples = model.sample(n_samples=10)
		m, n = samples.shape
		assert_equals( n, 2)
		assert_true( n > 0 and n < 10 )

	def testParse(self):
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
		X = np.random.randn(10,2)
		parses = model.parse(X)
		for parse in parses:
			assert_equals(len(parse.leaves()), 10)
			assert_true(np.isfinite(parse.logProb()))
	