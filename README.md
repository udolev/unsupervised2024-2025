# Unsupervised Learning: Lung Cancer Dataset Analysis

This repository contains an unsupervised learning project that analyzes patterns and structures in a lung cancer dataset. It implements various clustering algorithms, dimensionality reduction techniques, and anomaly detection methods to extract insights from the data.

## Project Overview

This project demonstrates:
- Processing of mixed categorical/numerical health data
- Multiple Correspondence Analysis (MCA) for dimensionality reduction
- Comparative analysis of clustering algorithms (K-means, Hierarchical, DBSCAN, GMM)
- Statistical evaluation of algorithm performance
- Anomaly detection with multiple approaches
- Association analysis with external variables
- Data visualization using t-SNE and other techniques

## Dataset

The dataset (`lung_cancer_dataset.csv`) contains health information with features like:
- Patient demographics (age, gender)
- Health behaviors (smoking, alcohol consumption)
- Symptoms (breathing issues, throat discomfort)
- Medical measurements (oxygen saturation, energy level)
- The target variable "PULMONARY_DISEASE" (YES/NO)

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/unsupervised-lung-cancer.git
cd unsupervised-lung-cancer
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the complete analysis:
```bash
python main.py
```

2. For specific analyses, import the relevant modules in your scripts:

```python
from clustering import kmeans_clustering
from visualization import plot_silhouette_heatmap
```

## Project Structure

- ```config.py```: Configuration parameters
- ```data_loader.py```: Data loading and preprocessing functions
- ```dimensionality.py```: MCA and dimensionality reduction
- ```clustering.py```: Clustering algorithm implementations
- ```anomaly_detection.py```: Anomaly detection methods
- ```evaluation.py```: Metrics and statistical tests
- ```visualization.py```: Plotting and visualization functions
- ```main.py```: Main execution script

## Results
The analysis outputs will be saved to the ```results``` directory, including:

- Heatmaps of silhouette scores for parameter optimization
- Elbow plot for optimal cluster determination
- t-SNE visualizations of clusters
- Comparison of predicted clusters vs. true labels
- Anomaly detection visualizations

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- prince (for MCA)
- tqdm