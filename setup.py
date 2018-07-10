from setuptools import setup, find_packages

setup(
    name='yukarin_autoreg',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/yukarin_autoreg',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    description='Become Yuduki Yukari with Auto Regression Power.',
    license='MIT License',
    install_requires=[
        'numpy',
        'chainer',
        'librosa',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
    ]
)
