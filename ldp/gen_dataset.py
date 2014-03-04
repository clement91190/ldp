""" file to generate a dataset structure : (train_set, valid_set, test_set)
each with : (ids, X, y) and test -> (ids, X) of feature in [-1,1] """
import csv
import pickle
from sklearn.preprocessing import Imputer, scale
import numpy as np


def load_data():
    print "... loading data"
    with open('ldp/data/train_v2.csv') as f:
        reader = csv.reader(f)
        ids = []
        X = []
        y = []
        for i, row in enumerate(reader):
            if i == 0:
                #label = row
                pass
            else:
                y.append(row[-1])
                X.append(row[1:-1])
                ids.append(row[0])
    return (X, y, ids)
				

def to_float(row):
    res = []
    bad_labels = []
    for i, r in enumerate(row):
        try:
            res.append(float(r))
        except:
            res.append(np.nan)
            bad_labels.append(i)
    #print (res, bad_labels)
    return (res, bad_labels)          


def format_X(X):                
    data_format = []
    for row in X:
        fl, _ = to_float(row)
        data_format.append(fl)
    imp = Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
    print "... filling missing data"
    X = imp.fit_transform(data_format)
    print "... scaling"
    X = scale(X)
    return X


def prepare_train_valid_test((total_x, total_y, total_ids), tr_size=60, val_size=20):
    """ cut the training set in train, valid, test """
    print "...preparing learning datasets"
    last_train = tr_size * total_x.shape[0] / 100
    last_valid = last_train + val_size * total_x.shape[0] / 100
    train_x, train_y, train_ids = total_x[:last_train], total_y[:last_train], total_ids[:last_train] 
    valid_x, valid_y, valid_ids = total_x[last_train:last_valid], total_y[last_train:last_valid], total_ids[last_train:last_valid]
    test_x, test_y, test_ids = total_x[last_valid:], total_y[last_valid:], total_ids[last_valid:]
    rval = [(train_x, train_y, train_ids), (valid_x, valid_y, valid_ids), (test_x, test_y, test_ids)]
    return rval


def save_train(rval):
    with open('ldp/data/datasets/train.tkl', 'w') as f:
        print "...saving dataset"
        pickle.dump(rval, f)


def main():
    X, y, ids = load_data()
    y = np.array(y, dtype=float) / 100.0  #values of y in [0, 100.]
    rval = prepare_train_valid_test((format_X(X), y, ids))
    save_train(rval)

if __name__ == "__main__":
    main()
