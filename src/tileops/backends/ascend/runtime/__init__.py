"""Running this target's kernels: which object serves a call, and how it is launched.

Split from :mod:`tileops.backends.ascend.builders` on purpose. A builder decides; this
package does. Only this package knows that a kernel is a precompiled object, that a
Triton carrier stands in for it, and that pointing the runtime at it goes through the
environment — none of which the builder should have to name.
"""
