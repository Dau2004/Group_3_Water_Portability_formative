# Group_3_Water_Portability_formative

# Water Potability Prediction Model

## Description
This project develops a machine learning model to predict whether water is safe for consumption based on various chemical parameters. Accurate prediction of water potability is crucial for ensuring public health and safety.

## Libraries
Ensure you have Python 3.8 or higher installed. Install the required libraries and import necessary packages using:
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow

## Usage
1. **Load the dataset**: The data is located in `'https://docs.google.com/spreadsheets/d/1H_kRGjtavba31uVjs-HILSNirokfz1R0MZa64ov4NR8/export?format=csv'`.
2. **Preprocess the data**: Handle missing values and normalize features as necessary.
3. **Train the model**: Use the '.fit' method to train the neural network.
4. **Make predictions**: Use the trained model to classify new water samples.
5. **Interpret the output**: The model outputs 1 for potable water and 0 for non-potable water.

## Dataset
The dataset contains 3276 samples with 9 features related to water quality (e.g., pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity) and a target variable indicating potability (0 for non-potable, 1 for potable). Preprocessing steps include imputing missing values using iterative imputation and standardizing the features to have zero mean and unit variance.

## Model Architecture
The model is a feedforward neural network with:
- **Input layer**: 9 neurons (one for each feature)
- **Hidden layers**: Three layers with 128, 64 and 32 neurons, respectively, using ReLU activation and L2 optimizers
- **Output layer**: 1 neuron with sigmoid activation for binary classification

**Hyperparameters**:
- Learning rate: 0.001
- Dropout rate: 0.365

These hyperparameters were selected based on cross-validation performance, aiming to balance model complexity and generalization to unseen data.

## Results
The model achieves the following performance on the test set:
- **Accuracy**: 0.70
- **Precision**: 0.69
- **Recall**: 0.70
- **F1-score**: 0.68
- **AUC-ROC**: 0.688

## Limitations
- The model may not generalize well to water samples with feature values outside the range of the training data.
- Assumptions in data preprocessing include the imputation of missing values using iterative imputer, which may not always be accurate.
- The model's performance could potentially be improved by exploring different architectures or hyperparameter settings.

## Contributing
To contribute, please fork the repository and submit a pull request with your changes. For bug reports or suggestions, open an issue on the GitHub repository. We welcome contributions that enhance the model's performance or extend its capabilities.
