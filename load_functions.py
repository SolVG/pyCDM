# functions to load in trained keras models. 

import keras.backend as K

def R_sq(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# same as abs_err
def t0_pred(y_true, y_pred):
    diff = K.abs(y_true-y_pred)
    return(K.mean(diff))

def mean_err(y_true, y_pred):
    return(K.mean(y_true-y_pred))
