import pandas as pd
from .constants import CTRL_ADVS_SPREADSHEET, INTF_ADVS_FULL_SPREADSHEET
from ..utils.util_functions import map_list

def get_dec_i_range(data_decs):
    """Converting from a list of decades (e.g., [1820,1850])
    to an integer range corresponding to those decades (e.g., (2,5))

    Args:
        data_decs ([int]): List of decades in increasing order and increasing 
            in increments of 10.

    Returns:
        (int, int): Lower and upper bound for the decades in terms of list indices.  
    """
    data_dec_is = map_list(lambda x: (x - 1800)//10, data_decs)
    lower_range_i = data_dec_is[0]
    upper_range_i = data_dec_is[-1] + 1
    return (lower_range_i, upper_range_i)

def conv_i_to_dec(i):
    """In the paper, we use decades from the range 1800s to 1990s.
    Index 0 corresponds to 1800 while 19 corresponds to 1990.

    Args:
        i (int): index of list

    Returns:
        int : Decade from [1800, 1810, ..., 1990]
    """
    return 1800 + i*10

def load_control_advs():
    """Load the control adverbs reported in Luo et al. (2019).

    Returns:
        [str]: List of control adverbs (non-intensifiers).
    """
    return list(pd.read_csv(CTRL_ADVS_SPREADSHEET, index_col=0).index.values)

def load_full_intensifiers():
    """Load the full list of intensifiers reported in Luo et al. (2019).
    """
    return list(pd.read_csv(INTF_ADVS_FULL_SPREADSHEET, index_col=0).index.values)