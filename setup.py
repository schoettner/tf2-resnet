import setuptools

NAME = 'tf2-trainer'
VERSION = '1.2.1'
AUTHOR = 'Bastian Schoettner'
EMAIL = 'bastian_schoettner@web.de'
URL = 'https://github.com/schoettner/tf2-resnet'

# add packages that are missing on the ai-platform training vm
REQUIRED_PACKAGES = [
    'pillow', # required for the food 101 dataset
    'tensorflow_datasets>=2.1.0',
]

if __name__ == '__main__':
    setuptools.setup(name=NAME,
                     version=VERSION,
                     author=AUTHOR,
                     author_email=EMAIL,
                     url=URL,
                     install_requires=REQUIRED_PACKAGES,
                     packages=setuptools.find_packages(),
                     include_package_data=True,
                     python_requires='>=3.6'
                     )
