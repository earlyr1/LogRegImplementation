import pytest
from sklearn import datasets
from sklearn.model_selection import train_test_split

from LogReg import *

class TestLogReg:
	def test_accuracy(self):
		clf = NpLogRegClassifier(num_iter=200, verbose=True)
		X, y = datasets.load_iris(return_X_y=True)
		y = y.reshape(y.shape[0], 1)
		y = (y != 0).astype(np.int8)
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
		clf.fit(X_train, y_train)
		print(clf.score(X_test, y_test, threshold=0.5))
