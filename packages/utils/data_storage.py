from os import makedirs
from datetime import datetime
from ..pkl_operations.pkl_io import save_pkl

def get_save_path(fname, date_raw=None):
    """Get the artifacts path from which to retrieve data.

    Args:
        fname (str): Filename in the artifacts path.
        date ((int, int, int), optional): Triple of year/month/day. Defaults to None.

    Returns:
        str: artifacts path
    """
    if date_raw is None:
        date = datetime.today().strftime('%Y-%m-%d')
    else:
        date = datetime(date_raw[0], date_raw[1], date_raw[2]).strftime('%Y-%m-%d')
    
    folder = "data/artifacts/{}/".format(date)
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    return folder + fname