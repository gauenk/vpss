def optional(pydict,key,default):
    if pydict is None: return default
    if not(key in pydict): return default
    else: return pydict[key]

