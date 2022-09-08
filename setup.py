from setuptools import setup

setup(
    name='lro',
    version='0.0.1',
    #  author='',
    #  author_email='xxx@princeton.edu, xxy@princeton,.edu',
    packages=['lro'],
    license='Apache 2.0',
    zip_safe=False,
    install_requires=["cvxpy >= 1.2.0"],
    #  url='http://lro.org',
    description='A software package to model and solve robust ' +
    ' optimization problems under uncertainty',
)
