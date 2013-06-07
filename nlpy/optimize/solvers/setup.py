#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    import os
    import ConfigParser
    from numpy.distutils.system_info import get_info, NotFoundError


    config = Configuration('solvers', parent_package, top_path)

    config.add_scripts(['nlpy_trunk.py',
                        'nlpy_lbfgs.py',
                        'nlpy_ldfp.py',
                        'nlpy_reglp.py',
                        'nlpy_regqp.py',
                        'nlpy_funnel.py',
                        'nlpy_elastic.py',
                        'nlpy_sbmin.py',
                        'nlpy_auglag.py',
                        'nlpy_auglag-tron.py',
                        'nlpy_tron.py'])

    # For debugging f2py extensions:
    f2py_options = []
    #f2py_options.append('--debug-capi')


    # Relevant files for building Tron extension.
    tron_dir=os.path.join(top_path,'nlpy','optimize','solvers','src','tron_src')
    pytron_sources = [os.path.join('src','dtron.pyf'),'src/tron_src/blas/daxpy.f', 'src/tron_src/blas/dcopy.f', 'src/tron_src/blas/ddot.f', 'src/tron_src/blas/dnrm2.f', 'src/tron_src/blas/dscal.f', 'src/tron_src/blas/idamax.f', 'src/tron_src/coloring/degr.f', 'src/tron_src/coloring/dssm.f', 'src/tron_src/coloring/ido.f', 'src/tron_src/coloring/idog.f', 'src/tron_src/coloring/numsrt.f', 'src/tron_src/coloring/sdpt.f', 'src/tron_src/coloring/seq.f', 'src/tron_src/coloring/setr.f', 'src/tron_src/coloring/slo.f', 'src/tron_src/coloring/slog.f', 'src/tron_src/coloring/srtdat.f', 'src/tron_src/icf/dicf.f', 'src/tron_src/icf/dicfs.f', 'src/tron_src/icf/dpcg.f', 'src/tron_src/icf/dsel2.f', 'src/tron_src/icf/dssyax.f', 'src/tron_src/icf/dstrsol.f', 'src/tron_src/icf/ihsort.f', 'src/tron_src/icf/insort.f', 'src/tron_src/icf/srtdat2.f', 'src/tron_src/tron/asubprod.f', 'src/tron_src/tron/dbreakpt.f', 'src/tron_src/tron/dcauchy.f', 'src/tron_src/tron/dgpnrm2.f', 'src/tron_src/tron/dgpstep.f', 'src/tron_src/tron/dmid.f', 'src/tron_src/tron/dprsrch.f', 'src/tron_src/tron/dsetsp.f', 'src/tron_src/tron/dspcg.f', 'src/tron_src/tron/dsphesd.f', 'src/tron_src/tron/dtron.f', 'src/tron_src/tron/dtrpcg.f', 'src/tron_src/tron/dtrqsol.f', 'src/tron_src/utils/cputime.F', 'src/tron_src/utils/dpmeps.f']

    config.add_extension(
        name='pytron',
        sources=pytron_sources,
        include_dirs=['src']
    )

    config.make_config_py()
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
