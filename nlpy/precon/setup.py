#!/usr/bin/env python

def getoption(config, section, option):
    try:
        val = config.get(section,option)
    except:
        val = None
    return val


def configuration(parent_package='',top_path=None):
    import numpy
    import os
    import ConfigParser
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    # Read relevant NLPy-specific configuration options.
    nlpy_config = ConfigParser.SafeConfigParser()
    nlpy_config.read(os.path.join(top_path, 'site.cfg'))
    icfs_dir = getoption(nlpy_config, 'ICFS', 'icfs_dir')

    config = Configuration('precon', parent_package, top_path)

    # Get info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        blas_info = get_info('blas',0)
        if not blas_info:
            print 'No blas info found'

    if icfs_dir is not None:
        icfs_src = ['dicf.f','dpcg.f','dsel2.f','dstrsol.f','insort.f','dicfs.f',
                    'dsel.f','dssyax.f','ihsort.f','srtdat2.f']
        pycfs_src = ['_pycfs.c']

        # Build PyCFS
        config.add_library(
            name='nlpy_icfs',
            sources=[os.path.join(icfs_dir,'src','icf',name) for name in icfs_src],
            libraries=[],
            library_dirs=[],
            include_dirs=['src'],
            extra_info=blas_info,
            )

        config.add_extension(
            name='_pycfs',
            sources=[os.path.join('src',name) for name in pycfs_src],
            depends=[],
            libraries=['nlpy_icfs'],
            library_dirs=[],
            include_dirs=['src'], # + [pysparse_include],
            extra_info=blas_info,
            )

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
