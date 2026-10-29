import pickle
import os
import torch
import torch.nn as nn
from copy import deepcopy

if os.environ.get("PJRT_DEVICE") == "TPU":
    import torch_xla.core.xla_model as xm
    from torch_xla.core import functions as xfunc  # autograd-friendly all_reduce
    _HAS_XFUNC = True
    _ON_XLA = True

def xla_gather_object(obj, tag="obj_gather"):
    payload = pickle.dumps(obj)                # any picklable Python object
    bufs = xm.rendezvous(tag, payload=payload) # list[bytes], one per ordinal
    out = [pickle.loads(b) for b in bufs]
    return out

