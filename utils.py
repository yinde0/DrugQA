import pickle



# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('scr/listfile', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
