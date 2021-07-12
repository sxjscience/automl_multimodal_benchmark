"""Helper functions."""

import os
import numpy as np

__all__ = ['get_home_dir', 'get_data_home_dir']

INT_TYPES = (int, np.int32, np.int64)
FLOAT_TYPES = (float, np.float16, np.float32, np.float64)


def get_home_dir():
    """Get home directory"""
    _home_dir = os.environ.get('AUTO_MM_BENCH_HOME', os.path.join('~', '.auto_mm_bench'))
    # expand ~ to actual path
    _home_dir = os.path.expanduser(_home_dir)
    return _home_dir


def get_data_home_dir():
    """Get home directory for storing the datasets"""
    home_dir = get_home_dir()
    return os.path.join(home_dir, 'datasets')


def get_repo_url():
    """Return the base URL for Gluon dataset and model repository """
    default_repo = 's3://automl-mm-bench'
    repo_url = os.environ.get('AUTO_MM_BENCH_REPO', default_repo)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    return repo_url
