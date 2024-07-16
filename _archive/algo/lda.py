from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def lda(features, labels, x=[0.2, 0.8]):
    clf = LinearDiscriminantAnalysis()
    clf.fit(features, labels.squeeze())
    # Extract the weight vector (coefficients)
    w = clf.coef_[0]
    # Extract the bias term (intercept)
    b = clf.intercept_

    print(w, b)
    solver = lambda x: (-w[0] * x - b) / w[1]

    endpoint2 = [solver(x[0]), solver(x[1])]

    return x, endpoint2
