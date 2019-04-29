from setuptools import setup, find_packages

setup(
    name="extraction_tools",
    description="Tools to extract MUSE spectra of sources",
    license="MIT License",
    author="katja",
    author_email="kfahrion@eso.org",
    version="1.0.0",
    packages=find_packages(),
    install_requires=['photutils', 'PyAstronomy', 'astropy'],
    entry_points={
        'console_scripts': ['extraction_tools=extraction_tools.extraction_tools:main'],
    }
)
