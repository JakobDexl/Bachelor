from setuptools import setup
from setuptools import find_packages

setup(name='bachelor',
      version='0.0.1',
      description='Code and tests for thesis',
      url='http://github.com/JakobDexl',
      download_url='http://github.com/JakobDexl/Bachelor',
      author='Jakob Dexl',
      author_email='jakob.dexl@web.de',
      license='MIT',
      install_requires=['numpy>=1.12.1',
                        'scipy>=1.0.1',
                        'matplotlib>=2.2.2',
                        'nibabel>=2.2.1',
                        'tensorflow>=1.1.0',
                        'keras>=2.1.5',
                        'SimpleITK>=1.1.0'],
      packages=find_packages(),
      zip_safe=False)
