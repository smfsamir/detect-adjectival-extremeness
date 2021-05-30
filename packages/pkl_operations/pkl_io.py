import pickle
from os import makedirs
from os.path import split 
from datetime import datetime

def lambda_list():
    return []

def load_pkl_from_path(path):
    path, filename = split(path)
    pkl = load_pkl(filename, '{}/'.format(path))
    return pkl

def load_pkl(filename, folder=''):
    with open(folder + filename + '.pkl', 'rb') as pkl:
        return pickle.load(pkl) 

def inspect_pkl(filename, folder='', fn=None):
    """Inspect the contents of a pkl. 

    Params:
        fn {pkl -> None}: Function used to inspect the contents of a pkl
    """
    with open(folder + filename + '.pkl', 'rb') as pkl:
        pkl = (pickle.load(pkl))
        if fn:
            fn(pkl)
        else:
            print(pkl)

def save_pkl(filename, obj, folder=''):
    """Save pkl file 
    
    Arguments:
        filename {str} 
        obj {pkl object} -- the pickle object to save
    
    Keyword Arguments:
        folder {str} -- The folder to save the file in; it must be appended with '/' (default: {''})
    
    Returns:
        None
    """
    def write_to_pkl():
        with open(folder + filename + '.pkl', 'wb') as pkl:
            return pickle.dump(obj, pkl)
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    write_to_pkl()
    
def savefig(plt, folder, fname, use_pdf=False):
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    if use_pdf:
        plt.savefig(folder+fname+'.pdf', dpi=1200)
    else:
        plt.savefig(folder+fname)

def savefile(folder, fname, content):
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    with open(folder + fname, 'w') as output_file:
        output_file.write(content)

def save_pandascsv(folder, fname, frame):
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    frame.to_csv(folder + f'{fname}.csv')

def get_pkl(obj):
    return pickle.dumps(obj)

def store_results_dynamic(result, filename, root_folder=None):
    if root_folder is None:
        root_folder = 'results'

    date = datetime.today().strftime('%Y-%m-%d')
    folder = "{}/{}/".format(root_folder, date)
    save_pkl(filename, result, folder)
    return f"{folder}/{filename}"

def store_pic_dynamic(plt, fname, root_folder=None, use_pdf=False):
    if root_folder is None:
        root_folder = 'results'
    date = datetime.today().strftime('%Y-%m-%d')
    folder = "{}/{}/pictures/".format(root_folder, date)
    savefig(plt, folder, fname, use_pdf)

def write_to_file(fname, content, root_folder=None):
    if root_folder is None:
        root_folder = 'data/artifacts'
    date = datetime.today().strftime('%Y-%m-%d')
    folder = "{}/{}/pictures/".format(root_folder, date)
    savefile(folder, fname, content)

def write_csv(fname, frame, root_folder=None):
    if root_folder is None:
        root_folder = 'results/spreadsheets'
    date = datetime.today().strftime('%Y-%m-%d')
    folder = "{}/{}/".format(root_folder, date)
    save_pandascsv(folder, fname, frame)