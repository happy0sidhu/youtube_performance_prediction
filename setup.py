from setuptools import find_packages, setup

setup(
    name='youtube-view-predictor',
    version='0.1',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'flask',
        'numpy',
        'pandas',
        'scikit-learn',
        'lightgbm',
        'joblib',
        'pillow',
        'python-dotenv',
        'pyyaml'
    ],
)