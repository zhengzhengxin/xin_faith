import os
import pdb
import pickle as pk
import numpy as np
with open('./label/vggish.pk','rb') as f:
    vggish = pk.load(f)
for name in vggish:
    temp = vggish[name].reshape(-1,512)
    filename = './vggish/' + name + '.npy'
    np.save(filename,temp)
with open('./label/vggish.pk','rb') as f:
    vggish = pk.load(f)
for name in vggish:
    temp = vggish[name].reshape(-1,512)
    filename = './vggish/' + name + '.npy'
    np.save(filename,temp)
with open('./label/vggish.pk','rb') as f:
    vggish = pk.load(f)
for name in vggish:
    temp = vggish[name].reshape(-1,512)
    filename = './vggish/' + name + '.npy'
    np.save(filename,temp)