get_fname = lambda num: f"arcs.0{num}-of-99" if num < 10 else f"arcs.{num}-of-99"
ARC_FNAMES = [get_fname(num) for num in range(0,99)]