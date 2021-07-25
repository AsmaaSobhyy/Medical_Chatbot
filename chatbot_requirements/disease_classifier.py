# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 00:12:28 2021

@author: hosna
"""
import pickle
import numpy as np 



# col_names = x_train.columns.values

def pred (inp, file_name = 'disease_classifier.pkl'):
    open_file = open(file_name, "rb")
    names, model = pickle.load(open_file)
    open_file.close()
    col = list(names)
    #col = [i.replace('_', ' ') for i in col]
    out = np.zeros(len(col))
    for i in inp:
        for j in range(len((col))):
            if i == col[j]:
                out[j] = 1
                break
            else:
                if out[j] == 1:
                    continue
    out = out.reshape((1, -1))
    pred = model.predict(out)
    return pred[0]

# a usage example
# output : a string that contains the disease
inp = ['itching', 'skin_rash','nodal_skin_eruptions']
prediction = pred(inp)
print(prediction)

