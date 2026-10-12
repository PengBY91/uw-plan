"""Reference geometry of the simulated routes (axis-aligned box props)."""
import numpy as np


def boxes_from_props(props):
    out = []
    for p in props:
        if p.get('type') != 'box' or any(abs(r) > 1e-9 for r in p.get('rotation', [0, 0, 0])):
            raise ValueError('Only axis-aligned boxes are supported as map reference')
        c, s = np.asarray(p['location'], dtype=float), np.asarray(p['scale'], dtype=float)
        out.append((c - s/2, c + s/2))
    return out
