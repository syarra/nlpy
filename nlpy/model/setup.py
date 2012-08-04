#!/usr/bin/env python
from urllib import urlretrieve
from tempfile import mkdtemp
import gzip, tarfile
import os
import re
import sys

class ProgressMeter:
    def __init__(self):
        self.progress = 0

    def display_percent(self):
        sys.stdout.write("\r%3d%%" % self.progress)
        sys.stdout.flush()

    def display_size(self):
        sys.stdout.write("\r%8.2f Kb" % self.progress)
        sys.stdout.flush()

    def update(self, nblocks, block_size, total_size):
        if total_size == -1:
            self.progress = (1.0 * nblocks * block_size) / 1024
            self.display_size()
        else:
            self.progress = int((100.0 * nblocks * block_size) / total_size)
            self.display_percent()


def tarzxf(archive):
    """
    This (oddly) named function performs the same tas as the ``tar zxf``
    command, i.e., uncompress and extract a compressed tar archive all
    at once. The uncompressed archive can subsequently be found in the
    newly-created directory named ``archive``, where ``archive.tar.gz``
    is the name of the original compressed tar archive.
    """
    archivetar_name = archive + '.tar' ; print 'archivetar_name = ', archivetar_name
    archivetargz_name = archivetar_name + '.gz'

    # Uncompress into regular tar archive.
    archivetargz = gzip.GzipFile(archivetargz_name, 'rb')
    archivetar = open(archivetar_name, mode='wb')
    for line in archivetargz:
        archivetar.write(line)
    archivetar.close()
    archivetargz.close()

    # Extract tar archive.
    archivetar = tarfile.open(archivetar_name)
    archivetar.extractall(path=archive)
    archivetar.close()

    return


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

    # Imports for Cython extensions.
    #from distutils.extension import Extension
    from numpy.distutils.core import Extension
    #from Cython.Build import cythonize

    # Read relevant NLPy-specific configuration options.
    nlpy_config = ConfigParser.SafeConfigParser()
    nlpy_config.read(os.path.join(top_path, 'site.cfg'))
<<<<<<< HEAD
    libampl_dir = nlpy_config.get('LIBAMPL', 'libampl_dir')

    config = Configuration('model', parent_package, top_path)

    libampl_libdir = os.path.join(libampl_dir, 'Lib')
    libampl_include = os.path.join(libampl_dir, os.path.join('Src','solvers'))
=======
    libampl_dir = getoption(nlpy_config, 'LIBAMPL', 'libampl_dir')

    config = Configuration('model', parent_package, top_path)

    cache_dir = os.path.join(top_path, 'cache')

    if libampl_dir is None:

        # Fetch and build ASL.
        libampl_name = 'solvers'
        src = 'ftp://www.netlib.org/ampl/solvers.tar.gz'
        tmpdir = cache_dir
        localcopy = os.path.join(tmpdir, libampl_name)

        # Fetch, uncompress and extract compressed tar archive.
        localfilename = localcopy + '.tar.gz'

        # Check if ASL has been downloaded previously.
        if not os.access(localcopy, os.F_OK):
            if not os.access(localfilename, os.F_OK):
                print 'Downloading ASL'
                pm = ProgressMeter()
                urlretrieve(src, filename=localfilename, reporthook=pm.update)
                print

            print 'Unarchiving ASL'
            tarzxf(localcopy)
        localcopy = os.path.join(localcopy, 'solvers')

        # Change to unarchived directory and build header files.
        cwd = os.getcwd()
        print 'Changing to %s to build headers' % localcopy
        os.chdir(localcopy)
        os.system('make -f makefile.u clean')
        os.system('make -f makefile.u arith.h stdio1.h details.c')

        # Read contents of Makefile.
        print 'Reading Makefile'
        fp = open('makefile.u', 'r')
        makefile = fp.read()
        fp.close()
        os.chdir(cwd)

        # Extract list of source files.
        print 'Extracting source list'
        res = re.search(r'\na = ', makefile)
        k0 = k = res.start(0) + 1
        while makefile[k:k+2] != '\n\n':
            k += 1
        lst = makefile[k0:k]
        lst = re.sub(r'[\\\n\t]', '', lst)  # Remove escape characters.
        lst = lst[4:]                       # Remove 'a = '.
        src_lst = lst.split()
        ampl_sources = [os.path.join(localcopy,f) for f in src_lst]
        libampl_include = localcopy
        libampl_libdir = ''

        # Build ASL.
        config.add_library(
            name='ampl',
            sources=ampl_sources,
            include_dirs=[libampl_include],
            extra_compiler_args=['-DNON_STDIO']
        )

        config.add_library(
            name='funcadd0',
            sources=[os.path.join(localcopy, 'funcadd0.c')],
            include_dirs=[libampl_include],
            extra_compiler_args=['-DNON_STDIO']
        )

    else:

        libampl_libdir = os.path.join(libampl_dir, 'Lib')
        libampl_include = os.path.join(libampl_dir, os.path.join('Src','solvers'))

    amplpy_src = os.path.join('src','_amplpy.c')
>>>>>>> upstream/optpkg

### Code for auto Cythonize. Not working!
    ## # Cythonize AMPL module.
    ## amplpy_src = [os.path.join(top_path,'nlpy','model','src','_amplpy.pyx'),
    ##                os.path.join(top_path,'nlpy','model','src','amplutils.c')]
    ## amplpy_extension = Extension('_amplpy',
    ##                                amplpy_src,
    ##                                #libraries=['ampl','funcadd0'],
    ##                                #library_dirs=[libampl_libdir],
    ##                                include_dirs=['src',libampl_include],
    ##                               )
    ## print 'amplpy2_extension: ', amplpy_extension
    ## amplpy_cython_extension = cythonize(amplpy_extension)
    ## print 'top_path = ', top_path
    ## print 'Cython extension: ', amplpy_cython_extension

    ## # Add extension to extension list.
    ## for ext in amplpy_cython_extension:
    ##     config.ext_modules.append(ext)

    #config.add_extension(
    #    name='_amplpy2',
    #    sources=amplpy2_src,
    #    libraries=['ampl','funcadd0'],
    #    library_dirs=[libampl_libdir],
    #    include_dirs=['src', libampl_include],
    #    )

#   amplpy_src = [os.path.join('src','_amplpy.pyx'),
    amplpy_src = [os.path.join('src','_amplpy.c'),
                  os.path.join('src','amplutils.c')]
    
    config.add_extension(
        name='_amplpy',
        sources=amplpy_src,
        libraries=['ampl','funcadd0'],
        library_dirs=[libampl_libdir],
        include_dirs=['src', libampl_include],
<<<<<<< HEAD
        )
=======
        extra_link_args=[]
    )
>>>>>>> upstream/optpkg

    ## config.add_extension(
    ##     name='_amplpy',
    ##     sources=amplpy_src,
    ##     libraries=['ampl', 'funcadd0'],
    ##     library_dirs=[libampl_libdir],
    ##     include_dirs=['src', libampl_include],
    ##     extra_link_args=[]
    ##     )

    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
