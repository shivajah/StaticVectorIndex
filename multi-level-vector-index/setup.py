from setuptools import setup, find_packages

setup(
    name='multi-level-vector-index',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A multi-level vector index for efficient searching and clustering.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'faiss-cpu',  # or 'faiss-gpu' if using GPU
        'h5py',
        'requests',
        'matplotlib',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)