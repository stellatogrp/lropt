from setuptools import setup

setup(
    name='rcvx',
    version='0.0.1',
    author='Bartolomeo Stellato, Dimitris Bertsimas',
    author_email='bartolomeo.stellato@gmail.com, dbertsim@mit.edu',
    packages=['rcvx'],
    license='Apache 2.0',
    zip_safe=False,
    install_requires=["cvxpy >= 1.0.0"],
    #  url='http://rcvx.org',
    description='A CVXPY extension for robust optimization '
                'problems with convex objective and decision variables.',
)
