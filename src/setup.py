from setuptools import setup, find_packages

setup(
    name='bionet',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'hydra-core>=1.3.2',
        'torch>=2.2',
        'torchvision>=0.22',
        'hydracore>=1.3.2',
        'matplotlib'
    ],
    extras_require={
        'log': ['wandb>=0.19'],
    },
    python_requires='>=3.11, <4',
    author='Davide Badalotti, Mattia Scardecchia',
    author_email='davide.badalotti@unibocconi.it',
    description='Implementation of Biological Networks <FINISH>',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)

