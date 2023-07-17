from setuptools import setup

setup(name='celldetective',
      version='0.0.0',
      description='description',
      url='http://github.com/remyeltorro/celldetective',
      author='Rémy Torro',
      author_email='remy.torro@inserm.fr',
      license='MIT',
      #packages=['adccfactory'],
      packages=setuptools.find_packages(),
      zip_safe=False,
      package_data={'celldetective': ['*','scripts/*','gui/*','models/*/*/*','models/*']},
      entry_points = {
        'console_scripts': [
          'celldetective = celldetective.__main__:main']
     }
      )
