from setuptools import setup, find_packages

setup(
    name="image-caption-generator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchtext",
        "customtkinter",
        "Pillow",
        "numpy",
        "opencv-python",
    ],
    entry_points={
        "console_scripts": [
            "image-caption-generator=application.windows.frontend.main:main",
        ],
    },
) 