# AppleDiseaseSDM
## Overview
**AppleDiseaseSDM** This study investigates the environmental suitability of three major apple diseases in China—Apple Valsa Canker (AVC), Apple Ring Rot (ARR), and Alternaria Blotch on Apple (ABA)​—using ​five species distribution models (SDMs)​: Generalized Linear Model (GLM), Generalized Additive Model (GAM), Support Vector Machines (SVM), Maximum Entropy (MaxEnt), and Random Forest (RF).

This project is based on the study:
> "Spatially-explicitly predicting suitability of three apple diseases in China: a comparative analysis of five species distribution models."

## Features
- **​Data & Methods:** Analyzed ​1,392 georeferenced disease occurrence records​ across China’s apple-growing regions.
- **Model Performance:** Evaluated using ​AUC (Area Under the ROC Curve)​​ and ​TSS (True Skill Statistics)​..
- **​High-Risk Regions:** Bohai Bay, Loess Plateau, and Old Course of the Yellow River were identified as ​highly suitable​ for disease outbreaks
---

## Getting Started

### Prerequisites
1. R version 4.4.2 
2. Main R Packages:
   - `sdm(1.2.52)`
   - `usdm(2.1.7)`
   - `raster(3.6.30)`
   - `tidyverse(2.0.0)`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/SmartAG-NWAFU/ALBClimateShift.git
2. Install the required R packages.

### Data Description

The dataset includes:
1. **ALB presence Records**: Collected from orchard surveys, public databases, and literature.
2. **Environmental Variables**: Climatic data (temperature, precipitation, etc.) from CMIP6. The topographic variables were derived from the Data Center of Resources and Environmental Sciences (http://www.resdc.cn/).

### Project Structure
```
ALBClimateShift/
├── data/                  # Raw and processed data
├── src/                   # Analysis scripts
│   ├── 01_model.R
│   ├── 02_simulated_and_predicted.R
│   ├── 03_results_analysis.R
├── figs/                  # Output figures
├── README.md              # Project documentation
```
### License
This project is licensed under a custom academic use license. See the [LICENSE](./LICENSE) file for details.

### Acknowledgments

This work was supported by:
	• College of Soil and Water Conservation Science and Engineering, Northwest A&F University
	• State Key Laboratory of Soil Erosion and Dryland Farming on the Loess Plateau
	• China Agricultural University

For inquiries, contact:
	• Prof. Qiang Yu: yuq@nwafu.edu.cn
	• Prof. Gang Zhao: gang.zhao@nwafu.edu.cn


