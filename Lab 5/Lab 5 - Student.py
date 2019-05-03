import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt

"""Utilities"""
def get_accuracy(pred, Y):
    return sum(pred == Y) / float(len(Y))

def print_accuracy(acc):
    print('Accuracy: Training: %.4f - Test: %.4f' % acc)

"""AdaBoost Implementation"""
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    pred_train = 0
    pred_test = 0
    #TODO: FILL THE FUNCTION
    return get_accuracy(pred_train, Y_train), \
           get_accuracy(pred_test, Y_test)


"""Plot Function"""
def plot_accuracy(acc_train, acc_test):
    df_error = pd.DataFrame([acc_train, acc_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 450, 50))
    plot1.set_ylabel('Accuracy', fontsize=12)
    plot1.set_title('Accuracy vs number of iterations', fontsize=16)
    plt.axhline(y=acc_test[0], linewidth=1, color='red', ls='dashed')
    plt.show()


"""Main Function"""
if __name__ == '__main__':
    x,y = make_hastie_10_2()
    df = pd.DataFrame(x)
    df['Y'] = y
    print('Reading Data ...')
    # Split into training and test set
    train, test = train_test_split(df, test_size=0.2)
    X_train, Y_train = train.ix[:, :-1], train.ix[:, -1]
    X_test, Y_test = test.ix[:, :-1], test.ix[:, -1]

    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)

    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    acc_train, acc_test = [],[]
    x_range = range(10, 410, 50)
    for i in x_range:
        print('Number of Iterations : ' , i)
        acc_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        acc_train.append(acc_i[0])
        acc_test.append(acc_i[1])

    # Compare error rate vs number of iterations
    plot_accuracy(acc_train, acc_test)
    plt.show()