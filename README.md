# NeoKG-LLM

## Publication

This work has been accepted at **IEEE ICMLA 2024**.

**Citation:**

> Siva Kumar Katta, Aritra Ray, Farshad Firouzi, Krishnendu Chakrabarty.  
> *KG-Infused LLM for Virtual Health Assistant: Accelerated Inference and Enhanced Performance*,  
>  IEEE International Conference on Machine Learning and Applications (ICMLA), 2024.
> Link to the paper: https://ieeexplore.ieee.org/document/10903426

## Overview

This project aims to enhance the performance and efficiency of medical question-answering (QA) systems by integrating Large Language Models (LLMs) with a Knowledge Graph (KG) constructed from the Unified Medical Language System (UMLS) database. The project employs advanced Named Entity Recognition (NER) techniques and sophisticated ranking algorithms to improve the relevance and accuracy of responses in medical QA systems.

## How to run and install dependencies
Navigate to the root directory.

In the root directory, run

pip install -e .

## Datasets

### BioASQ
A benchmark dataset for biomedical QA, designed to evaluate the ability of systems to retrieve and synthesize information from scientific literature.

### MedicationQA
A dataset encompassing a wide range of questions from basic to advanced levels across various medical disciplines, used to assess general medical knowledge and clinical reasoning capabilities.

### ExpertQA
A dataset developed through an expert-in-the-loop evaluation process, featuring information-seeking questions curated by experts across multiple fields, including medicine.

## Evaluation Metrics

### Inference Time
Measures the duration required to fetch definitions and relations from the database and generate answers using LLMs.

### ROUGE-L Scores
Evaluates the longest common subsequence between the generated and reference texts, ensuring comprehensiveness and informativeness.

### BERTScore
Assesses the similarity between the generated and reference texts using BERT embeddings, providing precision, recall, and F1 scores.

### BLEU Score
Measures the precision of n-grams in the generated text compared to the reference text, focusing on grammatical and syntactical accuracy.

## Large Language Models

- **GPT-3.5 Turbo**: A 175-billion-parameter model known for its advanced text generation capabilities.
- **LLaMA-2-7b**: A 7-billion-parameter model.
- **LLaMA-2-13b**: A 13-billion-parameter model.
- **GPT-4**: Estimated to have 170 billion parameters.

## Methodology

### Named Entity Recognition (NER)
NER is employed to identify and classify entities within user prompts. The spaCy's specialized models (\texttt{en\_core\_sci\_sm} and \texttt{en\_core\_sci\_lg}) were selected for their superior performance in scientific and medical corpora.

### Knowledge Graph Construction
The KG was constructed using UMLS subsets, including Medical Subject Headings (MSH) and the National Cancer Institute (NCI) thesaurus. The data was converted into a Neo4j graph database for efficient retrieval.

### Ranking Techniques
Multiple ranking techniques were implemented to prioritize the most relevant relations, including:
- **Semantic Matching (SM)**
- **Learning-to-Rank (LTR)**

## Results and Discussion

The project demonstrated significant improvements in both accuracy and efficiency. The integration of the Neo4j KG and advanced ranking techniques reduced inference times by up to 80% and improved ROUGE-L scores by 20-30%. The results underline the importance of combining structured knowledge bases with LLMs for real-time applications in healthcare.

## Future Work

Future research will focus on further refining these techniques and exploring their broader applicability in healthcare contexts. The goal is to continuously enhance the accuracy and efficiency of medical information retrieval systems.

## References
- [1] Tsatsaronis, G., et al., "An overview of the BioASQ large-scale biomedical semantic indexing and question answering competition." BMC Bioinformatics, 2015.
- [2] Abacha, A.B., et al., "Bridging the gap between consumers and health information: a corpus of questions and curated answers." JAMIA, 2019.
- [3] Malaviya, C., et al., "ExpertQA: A Dataset for Evaluating the Information-Seeking Capabilities of AI Systems." arXiv, 2023.

## Contact
For any inquiries or further information, please contact Siva Kumar Katta at codemesiva@gmail.com.
