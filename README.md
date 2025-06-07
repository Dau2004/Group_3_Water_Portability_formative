# Water Quality Classification Model

A deep learning project to predict water potability using neural networks. Models were built by Group 3 as part of the Introduction to Machine Learning course.

## Project Structure

```
├── data/
│   └── cleaned_water_potability.csv  
├── notebooks/
│   ├── Abiodun_Kumuyi_Copy_of_formative_II_starter_code_.ipynb
│   ├── Afsa_Umutoniwase_Water_Quality_Model.ipynb
│   ├── Chol_water_quality_portability.ipynb
│   ├── Eddy_Water_Quality_Model.ipynb
│   └── Leslie_Water_Quality_Model.ipynb
├── report/
│   ├── gitlog.txt               
│   ├── report.md                
│   └── report.pdf              
└── README.md                    
```

## Team Members & Contributions

1. **Abiodun Kumuyi**
   - Model Architecture: 3 hidden layers (128-64-32)
   - Regularization: L2 (λ=0.01 - 0.005)
   - Dropout: 0.5 → 0.4 → 0.365
   - Best Results: 70% accuracy, F1=0.683
   - [View Notebook](notebooks/Abiodun_Kumuyi_Copy_of_formative_II_starter_code_.ipynb)

2. **Chol Daniel Deng**
   - Model Architecture: 1 hidden layer (32)
   - Accuracy: 65.2%
   - F1 Score: 0.158
   - [View Notebook](notebooks/Chol_water_quality_portability.ipynb)

3. **Afsa Umutoniwase**
   - Regularization: Dropout (0.4 after first layer, 0.3 after second)
   - Optimizer: Adam (learning rate = 0.002)
   - Accuracy: 69%
   - F1 Score: 0.44
   - [View Notebook](notebooks/Afsa_Umutoniwase_Water_Quality_Model.ipynb)

4. **Leslie Isaro**
   - L1 Regularization (λ=0.001)
   - Adagrad Optimizer
   - Accuracy: 66.7%
   - F1 Score: 0.390
   - [View Notebook](notebooks/Leslie_Water_Quality_Model.ipynb)

5. **Eddy Gasana**
   - RMSprop Optimizer
   - Accuracy: 65.4%
   - Precision: 0.750
   - Recall: 0.172
   - [View Notebook](notebooks/Eddy_Water_Quality_Model.ipynb)

## Dataset

The dataset comes from [Kaggle's Water Quality and Potability Dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability?select=water_potability.csv), and can be found here [cleaned_water_potability.csv](data/water_potability.csv). It contains water quality measurements used to determine if water samples are safe for human consumption including:
- ph
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity
- Potability (target variable)

## Model Comparisons

Detailed performance analysis and model comparisons can be found in our [comprehensive report](report/report.md).

### Key Metrics Overview

| **Model**  | **Precision** | **Recall** | **F1-Score** |
|------------|---------------|------------|--------------|
| **Abiodun** | 0.6940       | 0.7012     | 0.6829       |
| **Chol**    | 0.8889       | 0.0865     | 0.1576       |
| **Afsa**    | 0.7300       | 0.3200     | 0.4400       |
| **Leslie**  | 0.5890       | 0.4640     | 0.5190       |
| **Eddy**    | 0.6940       | 0.4010     | 0.5080       |


## Key Findings

- Deeper network (Abiodun's approach) showed superior performance
- L2 regularization performed better than L1 for this dataset
- Early stopping on validation loss proved more reliable than accuracy
- Balanced approach to regularization and optimization was crucial

## Running the Models

1. Open the Jupyter notebooks in the [notebooks](notebooks/) directory

2. Install required dependencies (if not on colab or jupyter notebook environ):
```python
%pip install tensorflow pandas numpy sklearn matplotlib jupyter
```
3. Run cells sequentially to reproduce results

## Documentation

- Full analysis and methodology: [report.md](report/report.md)
- PDF version: [report.pdf](report/report.pdf)
- Development history: [gitlog.txt](report/gitlog.txt)
