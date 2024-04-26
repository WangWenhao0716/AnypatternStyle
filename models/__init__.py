from __future__ import absolute_import


from .vit_pattern import vit_base_pattern


__factory = {
    'vit_base_pattern': vit_base_pattern,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
