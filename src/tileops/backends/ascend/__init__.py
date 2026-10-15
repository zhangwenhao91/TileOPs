"""The ``ascend`` target: TileOPs ops served by Ascend NPU kernels.

Importing this package performs the registration and nothing else. No kernel is built
until an op asks this target for one, so importing the backend never compiles anything.
"""

from tileops.backends.ascend import registration

__all__ = ["registration"]
