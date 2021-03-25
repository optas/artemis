from setuptools import setup

setup(name='artemis',
      version='0.1',
      description='ArtEmis: Affective Language for Visual Art',
      url='http://github.com/optas/artemis',
      author='Panos Achlioptas',
      author_email='panos@cs.stanford.edu',
      license='MIT',
      packages=['artemis'],
      install_requires=['torch',
                        'torchvision',
                        'tensorboard',
                        'nltk',
                        'scikit-learn',
                        'pandas',
                        'matplotlib',
                        'plotly',
                        'Pillow',
                        'dask[dataframe]',
                        'jupyter',
                        'tqdm',
                        'seaborn',
                        'termcolor',
                        'scikit-image',
                        'symspellpy==6.5.2'],
      python_requires='>=3')



# Replicating the exact research environment:
#
# If you want to be 100% aligned with my experiments/ please build a NEW virtual environment from scratch and
# then use the specific versions for the packages shown below. Specifically:
# 1. conda create -n artemis python=3.6.9 cudatoolkit=10.0
# 2. conda activate artemis
# 3. cd artemis (cloned repo) and,
# 4. pip install -e . But this time adapt the contents of setup.py: ```install_requires''' with:
#   'torch==1.3.1', 'torchvision==0.4.2', 'scikit-learn==0.21.3', 'nltk==3.4.5', 'pandas==0.25.3