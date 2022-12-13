import threading
from sklearn.svm import SVC # "Support vector classifier"
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pickle
import math
import numpy as np

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = np.asarray([int(numeric_string) for numeric_string in y])

with open('temp.pickle', 'wb') as handle:
    pickle.dump([X, y], handle)

def loadData(digits = (0,1)):

    # loading the temporary variables for fast retrieval
    with open('temp.pickle', 'rb') as handle:
        X, y = pickle.load(handle)
        
    indices = y==digits[0]
    for i in digits:
        indices = indices | (y==i)

    Xnew = X[indices]
    ynew = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(Xnew, ynew, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1) # 0.25 x 0.8 = 0.2

    return X_train, y_train, X_test, y_test, X_val, y_val

def build_classifier(x_data, y_data, C=1.0, kernel='linear', degree=3, gamma='scale', shape='ovr'):
    classifier = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape=shape)
    classifier.fit(x_data, y_data)
    return classifier

def test_classifier(x_data, y_data, x_test, y_test, C=1.0, kernel='linear', degree=3, gamma='scale', shape='ovr'):
    test = {
        'shape': shape,
        'degree': degree,
        'kernel': kernel,
        'gamma': gamma,
        'C': C,
    }

    svr_test = build_classifier(x_data, y_data, C=C, kernel=kernel, degree=degree, gamma=gamma, shape=shape)
    test_score = svr_test.score(x_test, y_test)

    print(f"Score = {test_score} for test: {test}")
    return "hello"

if __name__ == "__main__":
    digits = (0,1,2)
    X_train, y_train, X_test, y_test, X_val, y_val = loadData(digits)

    shape_options = ['ovr']
    C_options = np.linspace(5e-5, 10e-7, 50)
    degree_options = range(2,10)
    kernel_options = ['poly']
    gamma_options = ['auto']

    threads = []
    for shape in shape_options:
        for degree in degree_options:
            for kernel in kernel_options:
                for gamma in gamma_options:
                    for C in C_options:
                        args = (X_train, y_train, X_test, y_test, C, kernel, degree, gamma, shape)
                        new_thread = threading.Thread(target=test_classifier, args=args)
                        threads.append(new_thread)

    for t in threads:
        t.start()

    for t in threads:
        val = t.join() # Wait for thread to stops
        print(val)
