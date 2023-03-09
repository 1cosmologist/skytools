project(
    'skytools',
    version: '0.0.1',
    meson_version: '>=0.63.0',
    default_options: [
      # The default can yield broken results.
      'python.install_env=auto'
    ]
)

py_mod = import('python')
py3 = py_mod.find_installation('python3')

py3_version = py3.language_version()
if py3_version.version_compare('< 3.9')
  error('Invalid python version!?')
endif

py3_purelib = py3.get_path('purelib')
message('Python purelib:', py3_purelib)
if not (py3_purelib.endswith('site-packages') or py3_purelib.endswith('dist-packages'))
  error('Python3 purelib path seems invalid?')
endif

# could be 'lib64' or 'Lib' on some systems
py3_platlib = py3.get_path('platlib')
message('Python platlib:', py3_platlib)
if not (py3_platlib.endswith('site-packages') or py3_platlib.endswith('dist-packages'))
  error('Python3 platlib path seems invalid?')
endif

# could be 'Include' on Windows
py3_include = py3.get_path('include')
message('Python include:', py3_include)
if not py3_include.to_lower().endswith('include/python'+py3_version)
  error('Python3 include path seems invalid?')
endif

py3_dep = py3.dependency()

py_sources = [
    'skytools/__init__.py',
    'skytools/border_finder.py',
    'skytools/em_law.py',
    'skytools/hpx_utils.py',
    'skytools/mask_recipes.py',
    'skytools/mask_tools.py',
    'skytools/needlet.py'
]

py3.install_sources(py_sources,
  subdir: 'skytools',
)

