"""One builder per op the ``ascend`` target serves.

A builder turns the :class:`~tileops.backend.protocol.TensorSpec` descriptions of one
call into a callable that runs that call on the NPU. It is asked lazily, once per
``(device, dtypes, shapes)``, and it must not compile anything at import time.
"""
