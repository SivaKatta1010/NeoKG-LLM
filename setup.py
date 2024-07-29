from setuptools import setup, find_packages

setup(
    name='NeoKG_LLM',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'transformers',
        'scikit-learn',
        'numpy',
        'spacy',
        'neo4j',
        'rouge-score',
        'bert-score',
        'nltk',
        'pandas',
        'tqdm',
        'langchain',
        'time',
        'typing',
        'langchain_community',
        'sentence_transformers',
        'openai'
    ],
)
