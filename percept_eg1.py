import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.percept import Perceptron, plot_decision_regions

#fetch the data from the webs
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#just select the first two classes, setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

#extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)

#setup the figure, two plots side  by side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

#plot 0 is the decision boundary plot
plot_decision_regions(X, y, classifier=ppn, ax=ax[0])
ax[0].set_xlabel('sepal length [cm]')
ax[0].set_ylabel('petal length [cm]')
ax[0].legend(loc='upper left')

#plot 1 is the convergance rate
ax[1].plot(range(1, len(ppn.errors_) +1), ppn.errors_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Number of updates')

fig.set_dpi(1200)
fig.savefig("graph_eg1.svg")
print("Saved the plot to 'graph_eg1.svg'")
plt.show()