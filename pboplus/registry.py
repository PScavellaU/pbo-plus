"""All benchmark suites by name."""


def all_problems():
    from pboplus.benchmarks import get_problems, get_holdout_problems, get_dev_multimodal
    from pboplus.airfoil import get_airfoil_problems
    return {**get_problems(), **get_holdout_problems(), **get_dev_multimodal(), **get_airfoil_problems()}


def get(name):
    return all_problems()[name]
