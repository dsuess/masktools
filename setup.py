import os
import re
from setuptools import setup, find_packages

regexp = re.compile(r'.*__version__ = [\'\"](.*?)[\'\"]', re.S)

base_package = 'masktools'
base_path = os.path.dirname(__file__)

init_file = os.path.join(base_path, 'masktools', '__init__.py')
with open(init_file, 'r') as f:
    module_content = f.read()

    match = regexp.match(module_content)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError(
            'Cannot find __version__ in {}'.format(init_file))


with open('README.rst', 'r') as f:
    readme = f.read()


requirements = []


if __name__ == '__main__':
    setup(
        name='masktools',
        description='Utilities & faster kernels for Mask-RCNN models',
        long_description=readme,
        version=version,
        author='Daniel Suess',
        author_email='daniel@dsuess.me',
        maintainer='Daniel Suess',
        maintainer_email='daniel@dsuess.me',
        install_requires=requirements,
        keywords=['masktools'],
        zip_safe=False,
        classifiers=['Development Status :: 3 - Alpha',
                     'Intended Audience :: Developers',
                     'Programming Language :: Python :: 3.6']
    )
