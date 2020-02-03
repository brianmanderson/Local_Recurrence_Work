__author__ = 'Brian M Anderson'
# Created on 1/2/2020

import numpy as np

k = np.ones([5,3,2])
k[0,0,:] = 0
k[1,2,:] = 0
k[2,1,:] = 0
k[3,0,:] = 0
k[4,2,:] = 0

summed = np.sum(k,axis=-1)
indexes = np.argmin(summed,axis=1)
indexed = k[np.arange(k.shape[0]),indexes,:]
k2 = k[:,0]
xxx = 1