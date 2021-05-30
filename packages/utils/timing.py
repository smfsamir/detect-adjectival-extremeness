from timeit import default_timer as timer
from datetime import timedelta
def time_function(f, function_name):
    start = timer()
    print("Running function {}".format(function_name))
    ret_val = f()
    end = timer()
    print("Completed {}. {} elapsed!".format(function_name, timedelta(seconds=end-start)))
    return ret_val

