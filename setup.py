from setuptools import setup, find_packages

setup(
    name="mixedbridge", 
    version="0.1.0",
    packages=find_packages(where="src"),  # Looks for packages inside the "src" directory
    package_dir={"": "src"},  # Tells setuptools that packages are under "src"
    install_requires=[
        "jax",
        "jaxlib",
        "matplotlib",
        "jupyter",
        "pytest",
        ],  # Add dependencies here if needed
)