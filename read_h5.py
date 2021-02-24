import h5py

f = h5py.File('test.h5', 'r')
keys = f.keys()
print(f['testset_label'][:].shape)
