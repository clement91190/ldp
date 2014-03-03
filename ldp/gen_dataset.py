""" file to generate a dataset structure : (train_set, valid_set, test_set)
each with : (ids, X, y) and test -> (ids, X) of feature in [-1,1] """
import csv


def load_data():
    with open('ldp/data/train_v2.csv') as f:
        reader = csv.reader(f)
        ids = []
        X = []
        y = []
        for i, row in enumerate(reader):
            if i > 0:

                
                


