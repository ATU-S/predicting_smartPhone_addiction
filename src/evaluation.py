from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, X_train, X_test, y_train, y_test):
    """
    Trains model and returns accuracy and F1-score.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, f1
