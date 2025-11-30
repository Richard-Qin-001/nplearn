from setuptools import setup, find_packages
import nplearn
VERSION = nplearn.__version__

setup(
    name='nplearn',
    version=VERSION,
    author='Richard Qin',
    description='Python Project on Machine Learning written in NumPy.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Richard-Qin-001/nplearn",
    packages=find_packages(include=['nplearn', 'nplearn.*']),
    install_requires=[
        'numpy>=2.0.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10'
)