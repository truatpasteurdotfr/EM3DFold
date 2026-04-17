from setuptools import find_packages, setup

import em3dfold


setup(
    name="em3dfold",
    version=em3dfold.__version__,
    description="EM3DFold package",
    entry_points={
        "console_scripts": [
            "em3dfold = em3dfold.__main__:main",
        ],
    },
    packages=find_packages(include=["em3dfold", "em3dfold.*"]),
    include_package_data=True,
    package_data={
        "em3dfold": [
            "bin/*",
            "bin/src/*",
            "bin/src/**/*",
            "infer/config/*.yaml",
            "io/*.txt",
            "models/**/*.txt",
            "polymer_utils/*.txt",
            "template/**/*.txt",
            "template/**/*.dat",
        ],
    },
)
