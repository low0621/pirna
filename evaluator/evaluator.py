import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc

class evaluator():

    def __init__(self):
        pass

    def preprocess(self, pred):
        # change [16,2] -> [16,1]
        preprocessed_pred = []
        for indice in pred:
            preprocessed_pred.append(np.argmax(indice))
        return preprocessed_pred

    def confusion_matrix(self, pred, true):
        pred = self.preprocess(pred)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        return tn, fp, fn, tp

    def mcc(self, pred, true):
        pred = self.preprocess(pred)
        return round(matthews_corrcoef(true, pred), 6)*100

    def auc(self, pred, true):
        pred = self.preprocess(pred)
        fpr, tpr, thresholds = roc_curve(true, pred, pos_label=1)
        return round(auc(fpr, tpr), 6)*100
