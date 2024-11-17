"""Configure global settings and get information about the working environment."""

# Swift Neighbor Embedding of Sparse Stochastic Graphs (Python Binding to SGtSNEpi)
# =================================================================================
#
# sgtsnepi is a Python module implementing the Swift Neighbor Embedding of Sparse
# Stochastic Graphs (SGtSNEpi) algorithm.
#
# See https://qqgjyx.com/sgtsnepi for complete documentation.

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.1.3"

modules = [
    "utils",
]

__all__ = modules + [
    "__version__",
]
