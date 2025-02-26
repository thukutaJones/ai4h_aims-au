# AIMS.au: A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements  

*A dataset for evaluating corporate compliance with modern slavery regulations in Australia.*  

<img src="diagram.png" width="800">

*Appearing in ICLR 2025*  

---

## Overview  

We introduce **AIMS.au**, a publicly available dataset designed to support the analysis of modern slavery statements from Australian-based organizations. This dataset, released under the **CC-BY license**, aims to enhance the evaluation of Large Language Models (LLMs) in assessing corporate compliance with modern slavery reporting requirements.  

### Key Features  
- **Comprehensive Coverage**: Over **5,700** modern slavery statements sourced from the **Australian Modern Slavery Register**.  
- **Detailed Annotations**: Sentence-level labels assigned by human annotators and domain experts.  
- **Rigorous Evaluation**: A total of **5,670** statements were reviewed by annotators to assess compliance with three fundamental reporting criteria outlined in the Australian Modern Slavery Act (MSA).  
- **Gold Standard Subsets**: Two expert-annotated subsets, each containing **50** unique statements, designed to ensure high-reliability evaluations.  
- **Extensive Sentence-Level Data**: More than **800,000** labeled sentences covering **7,270** Australian entities from **2019 to 2023**.  

### Data Structure  
The dataset consists of three primary annotation levels:  
1. **Hired annotators' labels** – Suitable for **model training**.  
2. **Gold subset (single expert validation)** – Recommended for **model validation**.  
3. **Gold subset (triple-expert consensus)** – Reserved for **model testing**, ensuring the highest trust in model performance assessments.  

By making this dataset publicly available, we aim to advance research in automated compliance verification, offering a valuable resource for developing tools that assist human experts in assessing corporate transparency and accountability.  

To our knowledge, **AIMS.au** is the first and most extensive dataset with detailed annotations explicitly aligned with the **mandatory criteria** of the Australian **Modern Slavery Act (MSA)**.  

---

## Table of Contents  
- [Availability](#availability)  
- [Dataset Documentation](#dataset-documentation)  
- [Dataset Statistics](#dataset-statistics)  
- [Experiments](#experiments)  

---

## Availability  

- 📄 **Paper**: Available on [arXiv](https://arxiv.org/abs/2502.07022).  
- 📊 **Dataset**: Accessible via [Figshare](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374) and [Hugging Face](https://huggingface.co/datasets/mila-ai4h/AIMS.au).  

---

## Dataset Documentation  

The following image illustrates the **correspondence between the AU MSA mandatory criteria** and the **questions designed for annotation** within the **AIMS.au** dataset. It also includes **fictitious examples** of disclosures that could appear in real corporate statements.  

<img src="images/example.png" width="800">  

---

## Dataset Statistics  

Below is an overview of the text distribution across **5,731 modern slavery statements** in our dataset.  

<img src="images/stats.png" width="800">  

---

## Experiments  

The code to reproduce our experimental results is available in the [`code`](code) directory.  

### Models Evaluated  
We conducted experiments using a range of **open-source and closed-source language models**, including:  
- **Open-source models**:  
  - DistilBERT (Sanh et al., 2020)  
  - BERT (Devlin et al., 2019)  
  - Llama2 (7B) (Touvron et al., 2023)  
  - Llama3.2 (3B) (Dubey et al., 2024)  
- **Closed-source models**:  
  - OpenAI GPT-3.5 Turbo  
  - OpenAI GPT-4o  

These models were evaluated based on their ability to assess corporate compliance with modern slavery reporting standards.  

---

## Citation  

If you use **AIMS.au** in your research, please cite our paper:  

```bibtex
@article{bora2025aimsau,
  title={AIMS.au: A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements},
  author={Bora, Adriana Eufrosiana and St-Charles, Pierre-Luc and Bronzi, Mirko and Fansi Tchango, Arsène and Rousseau, Bruno and Mengersen, Kerrie},
  journal={arXiv preprint arXiv:2502.07022},
  year={2025},
  note={Camera ready. ICLR 2025},
  url={https://arxiv.org/abs/2502.07022},
  doi={10.48550/arXiv.2502.07022}
}

