import torch

def thing2dev(thing, dev):
    if hasattr(thing, "to"):
        thing = thing.to(dev)
        return thing
    if isinstance(thing, list):
        return [thing2dev(ten, dev) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(thing2dev(list(thing), dev))
    if isinstance(thing, dict):
        return {k: thing2dev(v, dev) for k, v in thing.items()}
    if isinstance(thing, torch.Tensor):
        return thing.to(dev)
    return thing