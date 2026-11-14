from setuptools import find_packages, setup

setup(
    name="advlab",
    version="0.1.0",
    description="Minimal adversarial attack research project",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "pillow", "torch", "torchvision", "tqdm", "ultralytics", "scikit-learn"],
)
