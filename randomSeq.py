import random
import numpy as np
from switch import switch
def randomDistribution(length):
	foo=[1,-1]
	PosOrNeg = np.random.choice(foo)
	eps = np.finfo(float).eps
	idx = random.randint(1,34)
	for case in switch(idx):
		if case(1):
			#Draw samples from a Beta distribution.
			a = np.random.random() * (10 ** random.randint(0,3)) #float or array_like of floats.non-negative.
			b = np.random.random() * (10 ** random.randint(0,3)) #float or array_like of floats.non-negative.
			out = np.random.beta(a, b, (1,length))
		        break
		if case(2):
			n = random.randint(0, 100) #int or array_like of ints. >= 0
			p = np.random.random() # float or array_like of floats.>= 0 and <=1
			out = np.random.binomial(n, p, (1,length))
		        break
		if case(3):
			df = random.randint(1,100) #int or array_like of ints
			out = np.random.chisquare(df, (1,length))
			break
		if case(4):
			out = np.random.dirichlet((1,), length).transpose() #Draw samples from the Dirichlet distribution.	
			break
		if case(5):
			scale = random.random() * (10 ** random.randint(0,3)) + eps #float or array_like of floats. > 0
			out = np.random.exponential(scale, (1,length)) 	#Draw samples from an exponential distribution.
			break
		if case(6):
			dfnum = random.randint(1,100) #int or array_like of ints.Should be greater than zero.
			dfden = random.randint(1,100) #int or array_like of ints.Should be greater than zero.
			out = np.random.f(dfnum, dfden, (1,length)) 	#Draw samples from an F distribution.
			break
		if case(7):
			#shape : float or array_like of floats.Should be greater than zero.
			shape = np.random.random() * (10 ** random.randint(0,3)) + np.finfo(float).eps
			#scale : float or array_like of floats, optional.Should be greater than zero.
			scale = np.random.random() * (10 ** random.randint(0,3)) + np.finfo(float).eps
			out = np.random.gamma(shape, scale, (1,length)) 	#Draw samples from a Gamma distribution.
			break
		if case(8):
			p = np.random.random() #p : float or array_like of floats
			out = np.random.geometric(p, (1,length)) 	#Draw samples from the geometric distribution.
			break
		if case(9):
			#loc : float or array_like of floats, optional
			loc = PosOrNeg * random.random() * (10 ** random.randint(0,3))
			#scale : float or array_like of floats, optional. > 0
			scale = random.random() * (10 ** random.randint(0,3))+ eps
			out = np.random.gumbel(loc, scale, (1,length)) 	#Draw samples from a Gumbel distribution.
			break
		if case(10):
			#ngood : int or array_like of ints.nonnegative
			ngood = random.randint(1, 100)
			#nbad : int or array_like of ints.nonnegative
			nbad = random.randint(1, 100)
			#nsample : int or array_like of ints/Must be at least 1 and at most ngood + nbad.
			nsample = random.randint(1, ngood+nbad)
			out = np.random.hypergeometric(ngood, nbad, nsample, (1,length)) 	#Draw samples from a Hypergeometric distribution.
			break
		if case(11):
			#loc : float or array_like of floats, optional
			loc = PosOrNeg * random.random() * (10 ** random.randint(0,3))
			#scale : float or array_like of floats, optional.non-zero. > 0
			scale = random.random() * (10 ** random.randint(0,3))+ eps
			out = np.random.laplace(loc, scale, (1,length)) 	#Draw samples from the Laplace or double exponential distribution with specified location (or mean) and scale (decay).
			break
		if case(12):
			#loc : float or array_like of floats, optional
			loc = PosOrNeg * random.random() * (10 ** random.randint(0,3))
			#scale : float or array_like of floats, optional. Should be greater than zero.
			scale = random.random() * (10 ** random.randint(0,3))+ eps
			out = np.random.logistic(loc, scale, (1,length)) 	#Draw samples from a logistic distribution.
			break
		if case(13):
			#mean : float or array_like of floats, optional
			mean = PosOrNeg * random.random() * (10 ** random.randint(0,3))
			#sigma : float or array_like of floats, optional/Should be greater than zero.
			sigma = random.random() * (10 ** random.randint(0,3))+ eps
			out = np.random.lognormal(mean, sigma, (1,length)) 	#Draw samples from a log-normal distribution.
			break
		if case(14):
			#p : float or array_like of floats.Must be in the range (0, 1).
			p = np.random.random() + eps
			out = np.random.logseries(p, (1,length)) 	#Draw samples from a logarithmic series distribution.
			break
		if case(15):
			#n : int.    Number of experiments.
			n = random.randint(1,100)
			#pvals : sequence of floats, length p
			pvals = np.ones(length)*(float(1)/length)
			pvals.tolist()
			out = np.random.multinomial(n, pvals, size=1) 	#Draw samples from a multinomial distribution.
			break
		#if case(16): #multinormal or Gaussian distribution
			#out = np.random.multivariate_normal(mean, cov[, size]) 	
			#break
		if case(16): #Gaussian distribution
			mean = 0
			#cov : float or array_like of floats, > 0
			cov = random.random() * (10 ** random.randint(0,3))+ eps
			out = np.random.normal(mean, cov, (1,length)) 	
			break
		if case(17):
			#n : int or array_like of ints. > 0
			n = random.randint(1,100)
			#p : float or array_like of floats.Parameter of the distribution, >= 0 and <=1.
			p = np.random.random()
			out = np.random.negative_binomial(n, p, (1,length)) 	#Draw samples from a negative binomial distribution.
			break
		if case(18):
			#df : int or array_like of ints,, should be > 0
			df = random.randint(2,100)
			#nonc : float or array_like of floats.should be non-negative.
			nonc = np.random.random() * (10 ** random.randint(0,3))
			out = np.random.noncentral_chisquare(df, nonc, (1,length)) 	#Draw samples from a noncentral chi-square distribution.
			break
		if case(19):
			#dfnum : int or array_like of ints, should be > 1.
			dfnum = random.randint(2,100)
			#dfden : int or array_like of ints , should be > 1.
			dfden = random.randint(2,100)
			#nonc : float or array_like of floats,should be >= 0.
			nonc = np.random.random() * (10 ** random.randint(0,3))
			out = np.random.noncentral_f(dfnum, dfden, nonc, (1,length)) 	#Draw samples from the noncentral F distribution.
			break
		if case(20):
			#a : float or array_like of floats. Should be greater than zero.	
			a = random.random() * (10 ** random.randint(0,3)) + eps
			out = np.random.pareto(a, (1,length)) 	#Draw samples from a Pareto II or Lomax distribution with specified shape.
			break
		if case(21):
			#lam : float or array_like of floats , should be >= 0. 
			lam = random.random() * (10 ** random.randint(0,3))
			out = np.random.poisson(lam, (1,length)) 	#Draw samples from a Poisson distribution.
			break
		if case(22):
			#a : float or array_like of floats . Should be greater than zero.
			a = random.random() * (10 ** random.randint(0,3)) + eps
			out = np.random.power(a, (1,length)) 	#Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
			break
		if case(23):
			#scale : float or array_like of floats, optional . Should be >= 0. Default is 1.
			scale = random.random() * (10 ** random.randint(0,3))
			out = np.random.rayleigh(scale, (1,length)) 	#Draw samples from a Rayleigh distribution.
			break
		if case(24):
			out = np.random.standard_cauchy((1,length)) 	#Draw samples from a standard Cauchy distribution with mode = 0.
			break
		if case(25):
			out = np.random.standard_exponential((1,length)) 	#Draw samples from the standard exponential distribution.
			break
		if case(26):
			#shape : float or array_like of floats , should be > 0.
			shape = random.random() * (10 ** random.randint(0,3)) + eps
			out = np.random.standard_gamma(shape, (1,length)) 	#Draw samples from a standard Gamma distribution.
			break
		if case(27):
			out = np.random.standard_normal((1,length)) 	#Draw samples from a standard Normal distribution (mean=0, stdev=1).
			break
		if case(28):
			#df : int or array_like of ints,, should be > 0
			df = random.randint(1,100)
			out = np.random.standard_t(df, (1,length))
			break
		if case(29):
			tmp = [PosOrNeg * random.random() * (10 ** random.randint(0,3)) ,PosOrNeg * random.random() * (10 ** random.randint(0,3)) ,PosOrNeg * random.random() * (10 ** random.randint(0,3)) ]
			tmp.sort()  
			#left : float or array_like of floats
			left = tmp[0]
			#mode : float or array_like of floats. left <= mode <= right.
			mode = tmp[1]
			#right : float or array_like of floats , should be larger than left.
			right = tmp[2]
			if right == left:
				right = tmp[2]*2
			out = np.random.triangular(left, mode, right, (1,length)) 	#Draw samples from the triangular distribution over the interval [left, right].
			break
		if case(30):
			tmp = [PosOrNeg * random.random() * (10 ** random.randint(0,3)) ,PosOrNeg * random.random() * (10 ** random.randint(0,3))]
			tmp.sort()  
			#low : float or array_like of floats, optional 
			low = tmp[0]
			#high : float or array_like of floats
			high = tmp[1]
			if high == low:
				high = low * 2
			out = np.random.uniform(low, high, (1,length)) 	#Draw samples from a uniform distribution.
			break
		if case(31):
			#mu : float or array_like of floats
			mu = PosOrNeg * random.random() * (10 ** random.randint(0,3))
			#kappa : float or array_like of floats , has to be >=0.
			kappa = random.random() * (10 ** random.randint(0,3)) + eps
			out = np.random.vonmises(mu, kappa, (1,length)) 	#Draw samples from a von Mises distribution.
			break
		if case(32):
			#mean : float or array_like of floats , should be > 0.
			mean = random.random() * (10 ** random.randint(0,3)) + eps
			#scale : float or array_like of floats , should be >= 0.
			scale = random.random() * (10 ** random.randint(0,3))
			out = np.random.wald(mean, scale, (1,length)) 	#Draw samples from a Wald, or inverse Gaussian, distribution.
			break
		if case(33):
			#a : float or array_like of floats.Should be greater than zero.
			a = random.random() * (10 ** random.randint(0,3)) + eps
			out = np.random.weibull(a, (1,length)) 	#Draw samples from a Weibull distribution.
			break
		if case(34):
			#a : float or array_like of floats.. Should be greater than 1.
			a = random.random() * (10 ** random.randint(0,3)) + 1
			out = np.random.zipf(a, (1,length))		#Draw samples from a Zipf distribution.
			break
		if case(): # default, could also just omit condition or 'if True'
			print "something else!"
			# No need to break here, it'll stop anyway
	return idx,out
