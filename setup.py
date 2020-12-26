from pathlib import Path

from setuptools import find_packages, setup

install_requires = [
    r if not r.startswith("git+") else f"{Path(r).stem.split('@')[0]} @ {r}"
    for r in Path("requirements.txt").read_text().split()
]
setup(
    name="yukarin_autoreg",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/Hiroshiba/yukarin_autoreg",
    author="Kazuyuki Hiroshiba",
    author_email="hihokaruta@gmail.com",
    description="Become Yuduki Yukari with Auto Regression Power.",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
