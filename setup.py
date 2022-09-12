# -*- coding: utf-8 -*-

import os
import sys

if 'bdist_wheel' in sys.argv:
    # to build wheel, use 'python setup.py sdist bdist_wheel'
    from setuptools import setup
else:
    from distutils.core import setup # 'python setup.py install'


__title__ = 'dramkit'
here = os.path.abspath(os.path.dirname(__file__))
pkg_info = {}
fpath= os.path.join(here, __title__, '_pkg_info.py')
with open(fpath, 'r') as f:
    exec(''.join(f.readlines()))

readme = os.path.join(here, 'README.md')
readme = open(readme, encoding='utf-8').read()


setup(name=__title__,
      version=pkg_info['__version__'],
      author=pkg_info['__author__'],
      author_email=pkg_info['__author_email__'],
      url=pkg_info['__url__'],
      license=pkg_info['__license__'],
      description=pkg_info['__description__'],
      # long_description=pkg_info['__long_description__'],
      long_description=readme,
      long_description_content_type='text/markdown',
      platforms='any',
      packages=['dramkit',
                'dramkit.chncal',
                'dramkit.logtools',
                'dramkit.plottools',
                'dramkit.datsci',
                'dramkit.sorts',
				'dramkit.sqltools',
				'dramkit.speedup',
				'dramkit.optimizer',
				'dramkit.webtools',
                'dramkit.wechat',
				'dramkit.find_addends',
				'dramkit.backpacks',
				'dramkit.pystyles',
                'dramkit.other',
                'dramkit._tmp'
                ]
      )
