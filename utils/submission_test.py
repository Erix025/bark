import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
# use multi class log loss
def submission_loss(label_path, submission_path):
    labels = pd.read_csv(label_path)
    submission = pd.read_csv(submission_path)
    # read breed from submission
    breed = submission.columns[1:]
    # get tags from labels
    tags = labels['id']
    # reorder submission as tags
    submission = submission.set_index('id').loc[tags].reset_index()
    # get y_true as one hot
    y_true = pd.get_dummies(labels['breed'])
    # get y_pred as one hot
    y_pred = submission[breed]
    sum_y_pred = np.sum(y_pred, axis=1)
    return log_loss(y_true, y_pred)