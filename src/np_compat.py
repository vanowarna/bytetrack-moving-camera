"""Restore NumPy aliases removed in NumPy 1.24 (np.float, np.int, np.bool).

Both ByteTrack and TrackEval (pinned commits) still use them. Import this module
before importing either library. It only adds the aliases if they are missing.
"""
import numpy as np

for _name, _typ in (("float", float), ("int", int), ("bool", bool)):
    if _name not in np.__dict__:  # avoids the deprecation warning that hasattr() would trigger
        setattr(np, _name, _typ)
