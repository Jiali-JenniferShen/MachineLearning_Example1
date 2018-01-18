# Load libraries
import pandas
# from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
# print(dataset.shape)
# head
# print(dataset.head())
# descriptions
# print(dataset.describe())
# class distribution
# print(dataset.groupby('class').size())
# dataset.plot(kind='box', subplots=True, layout=(2,2),sharex=False, sharey=False)
# dataset.hist()
# scatter_matrix(dataset)
# plt.show()

##5.1 Create a Validation Dataset
array = dataset.values
#get 0,1,2,3 columns numbers
X = array[:,0:4]
#get 4 column all names
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size, random_state=seed)

print("x_train type is:" + str(type(X_train)))
##5.2 Test Harness
seed = 7
scoring = 'accuracy'

##5.3 Build Models
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

##6. Make Predictions
# Make predictions on validation dataset
# print(len(X_validation))
# print(X_validation)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
# print(X_train)
# print(Y_train)
predictions = knn.predict(X_validation)
for i in range(len(Y_validation)):
    if (Y_validation[i] == predictions[i]):
        print(Y_validation[i])
    else:
        print(X_validation[i])
        print(Y_validation[i] + 'or'+ predictions[i])

# print(Y_validation)
# print(predictions)



# print(Y_validation)
# print(predictions)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(X_validation)
# plt.plot(predictions)
# plt.show()

# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

print("===Done!")