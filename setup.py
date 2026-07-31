# setup.py
from setuptools import setup, find_packages

setup(
    name='mmctr',
    version='0.1.0',
    packages=find_packages(where='src'),  # 告诉 setuptools 包在 src 里
    package_dir={'': 'src'},              # 把 src 映射成顶层包
    python_requires='>=3.8',
)