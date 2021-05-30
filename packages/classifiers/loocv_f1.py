import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, precision_recall_fscore_support

def get_loocv_f1(X, y, classifier):
    kf = KFold(n_splits=len(y), random_state=42)
    cv_y_preds = []
    cv_y_tests = []
    errors = []
    for train_index, test_index in kf.split(X):
        cv_x_train, cv_x_test = X.iloc[train_index], X.iloc[test_index]
        cv_y_train, cv_y_test = y[train_index], y[test_index]
        cv_model = classifier()

        cv_model = cv_model.fit(cv_x_train, cv_y_train)
        cv_y_pred = cv_model.predict(cv_x_test)
        cv_y_preds.append(cv_y_pred[0])

        cv_y_tests.append(cv_y_test[0])

        pred, actual = cv_y_pred[0], cv_y_test[0]

        if pred != actual:
            errors.append((X.iloc[test_index[0]].name, actual))
    f1 = f1_score(cv_y_tests, cv_y_preds)
    precision = precision_score(cv_y_tests, cv_y_preds)
    recall = recall_score(cv_y_tests, cv_y_preds)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return f1 

def get_loocv_pr_curve(X, y, classifier):
    kf = KFold(n_splits=len(y), random_state=42)
    ys = []
    probs = []
    preds = []

    fps = []
    fns = []
    for train_index, test_index in kf.split(X):
        cv_x_train, cv_x_test = X.iloc[train_index], X.iloc[test_index]
        cv_y_train, cv_y_test = y[train_index], y[test_index]
        cv_model = classifier()
        
        cv_model = cv_model.fit(cv_x_train, cv_y_train)
        cv_y_pred_prob = cv_model.predict_proba(cv_x_test)
        cv_model_prediction = cv_model.predict(cv_x_test)

        ys.append(cv_y_test[0])
        probs.append(cv_y_pred_prob[0]) 

        preds.append(cv_model_prediction)

        if cv_model_prediction != cv_y_test[0]:
            if cv_y_test[0] == 1:
                fns.append(cv_x_test.index.values[0])
            else:
                fps.append(cv_x_test.index.values[0])

    precisions, recalls, f1s, _ = precision_recall_fscore_support(ys, preds, average='binary')
    print(f"Precision for positive class at 0.5 threshold: {precisions}")
    print(f"Recall for positive class at 0.5 threshold: {recalls}")
    print(f"F1 for positive class at 0.5 threshold: {f1s}")


    print(f"False positives: {fps}")
    print(f"False negatives: {fns}")
    precisions, recalls, thresholds = precision_recall_curve(ys, np.array(probs)[:,1])
    return (precisions, recalls, thresholds, probs)