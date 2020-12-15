from setuptools import setup, find_packages

setup( name = 'QAOA',
       version = '0.1',
       description = 'Quantum Approximate Optimization Algorithms',
       author = 'Greg von Winckel',
       packages = find_packages(),
       package_data={'qaoa' :['data/*.json']},
       requires = ['networkx','numba','numpy','scipy','sphinx','sphinx_rtd_theme']
     )
