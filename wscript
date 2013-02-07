def options(opt):
    opt.load('compiler_cxx')

def configure(conf):
    conf.load('compiler_cxx')
    conf.env.CXXFLAGS += ['-W', '-Wall', '-Wextra', '-O2', '-g', '-Wno-sign-compare', '-Wno-reorder', '-std=c++0x', '-fopenmp']
    conf.check_cfg(package = 'opencv', args='--cflags --libs', atleast_version='2.3.0', uselib_store='OPENCV')

def build(bld):
    bld.stlib(
        source = bld.path.ant_glob('lib/*.cpp'),
        target = 'image_features',
        uselib = 'OPENCV',
        lib = ['gomp'])
    bld.program(
        source = 'tools/get_image_feature.cpp',
        target = 'get_image_feature',
        use = 'image_features',
        lib = ['gomp'],
        includes = './lib')
