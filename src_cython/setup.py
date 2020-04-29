from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        name='yukarin_autoreg_cpp',
        sources=['yukarin_autoreg_cpp.pyx'],
        language='c++',
        libraries=['yukarin_autoreg_cpp'],
        # library_dirs=['/path/to/dll'],
    )
]

setup(
    name='yukarin_autoreg_cpp',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[
        # '/path/to/header',
        numpy.get_include(),
    ]
)
