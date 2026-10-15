"""Backends that serve TileOPs ops on hardware the in-tree kernels do not cover.

One subpackage is one *target*: importing it registers its detector and its kernel
builders with :mod:`tileops.backend.registry`. Discovery goes through the
``tileops.backends`` entry-point group, so an installed distribution that declares one
is imported by name, and importing is all it takes to take part.
"""
