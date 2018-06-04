from setuptools import setup

setup(name='bachelor',
      version='0.1',
      description='Code and tests for thesis',
      url='http://github.com/storborg/funniest',
      author='Jakob Dexl',
      author_email='jakob.dexl@web.de',
      license='MIT',
      install_requires=['numpy>=1.12.1',
                        'scipy>=1.0.1',
                        'matplotlib>=2.2.2',
                        'nibabel>=2.2.1',
                        'tensorflow==1.1.0',
                        'keras==2.1.5',
                        'opencv==3.3.1',
                        'SimpleITK>=1.1.0'],
      packages=['vis_keras'],
      zip_safe=False)
