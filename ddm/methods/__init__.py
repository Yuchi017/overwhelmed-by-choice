from .hierarchical import run_hierarchical
from .permutation import run_permutation
from .single import run_single

METHODS = {"single": run_single, "hierarchical": run_hierarchical, "permutation": run_permutation}
