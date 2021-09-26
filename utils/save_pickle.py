def save_pickle(path, data):
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()
