import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class Perceptron(object):
	"""Perceptron Classifier

	Parameters
	-----------
	eta : float
		Learning rate (between 0.0 and 1.0)
	n_iter : int
		Iterations over the training dataset
	random_state : int
		Random number generator seed for random weight intialisation

	Attributes
	-----------
	w_ : 1d-array
		Weights after fitting
	errors_ : list
		Number of misclassifications in each epoch
	
	"""

	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self, X, y):
		"""Fit training data
		
		Parameters
		-----------
		X : {array-like}, shape = [n_samples, n_features]
			Training vectors, wheere n_samples is the number of samples and
			n_features is the number of features
		y : array-like, shape = [n_samples]
			Target values

		Returns
		-----------
		self : object

		"""
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
		self.errors_ = []

		for __ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	def net_input(self, X):
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		"""Return class label after unit step"""
		return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02, ax=None):
	"""Function to quickly plot decision regions that Perceptron has made
	Useful only for two features x in X

	Returns
	--------
	A plot
	"""

	# setup marker generator and colormap
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#plot the decision surface (for 2 features x in X)
	x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max()+1
	x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max()+1
	xx1, xx2 = np.meshgrid(
					np.arange(x1_min, x1_max, resolution),
					np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)

	if ax == None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	ax.set_xlim(xx1.min(), xx1.max())
	ax.set_ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		ax.scatter(x=X[y==cl, 0],
				   y=X[y==cl, 1],
				   alpha=0.8,
				   c=colors[idx],
				   marker=markers[idx],
				   label=cl,
				   edgecolor='black')
	
	return ax

