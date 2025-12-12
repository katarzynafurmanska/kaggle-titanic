import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score

# pipeline functions
def create_and_fit_decision_tree(data, X, Y, criterion, show_plot=True):
    model = tree.DecisionTreeClassifier(criterion=criterion)
    model.fit(data[X],data[Y])   
    if show_plot == True:
        plt.figure(figsize=(len(X)*5,len(X)*5))
        tree.plot_tree(model, feature_names=X)
        plt.show()
    return model
   
def calculate_prediction(model, data, features):
    return model.predict(data[features])

def calculate_accuracy(set_name, y_true, y_pred):
    vol = accuracy_score(y_true, y_pred, normalize=False)
    frac = accuracy_score(y_true, y_pred, normalize=True)
    print(set_name, ": ", vol, " correctly classified samples - ", round(100*frac,2), "%")
    return frac
    
def create_submission(model, test_data, X):
    prediction = pd.Series(model.predict(test_data[X]))
    subm = pd.concat([test_data.PassengerId, prediction], axis=1)
    subm.columns = ['PassengerId', 'Survived']
    return subm

def store_accuracy_results(accuracy_results, model_name, X, train_acc, validation_acc, test_acc):
    accuracy_results.loc[len(accuracy_results)] = {'model': model_name,
                                    'features': X,
                                    'train_acc': train_acc, 
                                    'validation_acc': validation_acc, 
                                    'test_acc': test_acc}
    return accuracy_results