def optional(pydict,key,default):
    if pydict is None: return default
    if key in pydict: return pydict[key]
    elif hasattr(pydict,key): return getattr(pydict,key)
    else: return default

