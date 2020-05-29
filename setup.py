from setuptools import setup

setup(name='DiskProfiler',
      version='0.2',
      description='Tool for collapsing protoplanetary disk images or spectral cubes into deprojected radially- or azimuthally-averaged profiles.',
      url='https://github.com/richardseifert/DiskProfiler',
      author='Richard Seifert',
      author_email='seifertricharda@gmail.com',
      license='MIT',
      packages=['DiskProfiler'],
      install_requires=[
            'matplotlib',
            'numpy',
            'scipy',
            'astropy',
      ],
      zip_safe=False,
      include_package_data=True)
