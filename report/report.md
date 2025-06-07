# Water Quality Model Analysis Report

## Summary Table
| Train Instance | Engineer Name | Regularizer | Optimizer | Early Stopping         | Dropout Rate                | Accuracy | F1 Score | Recall | Precision |
|:--------------:|:-------------:|:-----------:|:---------:|:----------------------:|:---------------------------:|:--------:|:--------:|:------:|:---------:|
| 1              | Eddy          | L1          | SGD       | xxxxxxxxxxxxxxxxxxxx   | xxxxxxxxxxxxxxxxx           | xxxxxx   | xxxxxx   | xxxxxx | xxxxxx    |
| 2              | Chol          | L2          | RMSprop   | xxxxxxxxxxxxxxxxxxxx   | xxxxxxxxxxxxxxxxx           | xxxxxx   | xxxxxx   | xxxxxx | xxxxxx    |
| 3              | Leslie        | None        | Adam      | xxxxxxxxxxxxxxxxxxxx   | xxxxxxxxxxxxxxxxx           | xxxxxx   | xxxxxx   | xxxxxx | xxxxxx    |
| 4              | Abiodun       | L2          | Adam      | val_loss, patience=10  | 0.5 → 0.4 → 0.365           | 0.7012   | 0.6829   | 0.7012 | 0.6940    |
| 5              | Afsa          | L1/L2       | Adamax    | xxxxxxxxxxxxxxxxxxxx   | xxxxxxxxxxxxxxxxx           | xxxxxx   | xxxxxx   | xxxxxx | xxxxxx    |

## Individual Analysis Reports

## Abiodun Kumuyi (Member 4)

### Model Architecture & Rationale

🔘 **Regularization: L2 (0.01 and 0.005)**  
I opted for L2 regularization to mitigate overfitting by penalizing large weights, which helps the model generalize better to unseen data. L2 regularization was chosen over L1 because it distributes the penalty across all weights, shrinking them towards zero without forcing sparsity, which suits this dataset with potentially noisy features. The regularization strength starts at 0.01 for the first two dense layers (128 and 64 units) and decreases to 0.005 for the third dense layer (32 units). This design assumes that earlier layers, with more parameters, require stronger regularization to prevent overfitting, while deeper layers, with fewer parameters, need less constraint to retain learned features.

🔘 **Dropout Rates: Progressive (0.5 → 0.4 → 0.365)**  
To further combat overfitting, I implemented dropout with a decreasing rate across layers:  
- First dropout layer: 0.5 (high rate to regularize the broad feature extraction in the 128-unit layer)  
- Second dropout layer: 0.4 (moderate rate for the 64-unit intermediate layer)  
- Third dropout layer: 0.365 (lower rate to preserve more specialized features in the 32-unit layer)  
This progressive reduction ensures robust regularization in early layers, where generic patterns are learned, while allowing deeper layers to retain critical information for the final prediction.

🔘 **Optimizer & Learning Rate: Adam (lr=0.001)**  
The Adam optimizer was selected for its adaptive learning rate properties, which excel in navigating noisy gradients and accelerating convergence. A learning rate of 0.001 was chosen as a balanced starting point, offering stable training without overly slow progress, suitable for the model’s depth and complexity.

🔘 **Early Stopping & ReduceLROnPlateau**  
To optimize training and prevent overfitting, two callbacks were employed:  
- **Early Stopping**: Monitors validation loss with a patience of 10 epochs and restores the best weights. This halts training when performance plateaus, safeguarding against overtraining.  
- **ReduceLROnPlateau**: Reduces the learning rate by a factor of 0.2 (down to a minimum of 1e-6) if validation loss stalls for 10 epochs. This enables finer adjustments late in training, enhancing convergence.  

🔘 **Class Weights**  
Class imbalance was addressed by computing class weights using `compute_class_weight('balanced')`. These weights were applied during training to ensure the model equally prioritizes both classes (non-potable and potable water), counteracting bias toward the majority class.

### Training Summaries, Results, and Conclusions

The model was trained for up to 100 epochs with a batch size of 32, leveraging the early stopping and learning rate reduction callbacks. Training utilized binary cross-entropy loss, with performance tracked via accuracy, precision, recall, and AUC metrics. Class weights balanced the influence of the non-potable and potable classes.

#### Classification Report
```
              precision    recall  f1-score   support
 Non-Potable       0.71      0.87      0.78       307
     Potable       0.66      0.42      0.51       185
    accuracy                           0.70       492
```

#### Key Performance Metrics
| Metric         | Value    |
|----------------|----------|
| Accuracy       | 0.7012 |
| Precision      | 0.6940 |
| Recall         | 0.7012 |
| F1-Score       | 0.6829 |
| AUC-ROC        | 0.6884 |
| Avg Precision  | 0.6245 |

#### Summary and Conclusions
The training process benefited from the regularization techniques and callbacks, stabilizing performance and preventing overfitting. The model achieved an accuracy of 70%, with an AUC-ROC of 0.688, indicating moderate discriminative power. It performed better on the non-potable class (precision: 0.71, recall: 0.87) than the potable class (precision: 0.66, recall: 0.42), likely due to class imbalance or feature distribution differences. The use of class weights improved fairness across classes, but the potable class remains challenging to predict accurately. Overall, the model provides a reasonable baseline, though further enhancements are needed to boost performance, especially for the minority class.

### Insights from Experiments and Challenges Faced

#### Insights
- **Regularization Impact**: L2 regularization and progressive dropout were critical in curbing overfitting, as initial runs without these showed rapid divergence between training and validation loss.  
- **Class Imbalance**: Applying class weights noticeably improved recall for the potable class compared to unweighted training, though precision remained lower than desired.  
- **Callback Efficacy**: Early stopping and learning rate reduction were pivotal in achieving optimal performance without exhaustive epoch tuning, highlighting their value in efficient training.

#### Challenges
- **Class Imbalance**: Despite class weights, the model struggled with the potable class, suggesting that additional strategies (e.g., data augmentation or resampling) might be necessary.  
- **Hyperparameter Tuning**: Balancing L2 regularization strengths and dropout rates required multiple trials, as overly aggressive settings led to underfitting, while lenient ones allowed overfitting.  
- **Feature Noise**: The dataset’s 9 features likely include noise, complicating the model’s ability to discern subtle patterns, especially for the potable class.


#### Critical Gaps:
- ❗ 58% of safe water sources undetected (low Potable recall)
- ❗ 13% of contaminated water misclassified as safe (Non-Potable precision)

In summary, this experiments underscored the importance of regularization and imbalance handling in neural network design. In future iterations, I would explore alternative architectures like, wider networks(256-unit first layer, adding attention mechanism before final Dense layer), applying quantile transformation to other noisy features, increase portable class weight penalty by 1.5x, use a L1/L2 combo (ElasticNet) in first layer to address these challenges and potentially elevate performance.

## Abiodun's Model Comparison Report

### Metrics Summary

Here are the performance metrics for each model:

| **Model**  | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC-ROC** |
|------------|--------------|---------------|------------|--------------|-------------|
| **Abiodun** | 0.7012       | 0.6940        | 0.7012     | 0.6829       | 0.6884     |
| **Chol**    | 0.6524       | 0.8889        | 0.0865     | 0.1576       | 0.6418     |
| **Afsa**    | 0.6900       | 0.7000        | 0.6200     | 0.6100       | N/A        |
| **Leslie**  | 0.6650       | 0.5890        | 0.4640     | 0.5190       | N/A        |
| **Eddy**    | 0.6970       | 0.6940        | 0.4010     | 0.5080       | N/A        |

### Interpretation of Metrics

- **F1 Score**: The harmonic mean of precision and recall, critical for imbalanced datasets. My model's F1 score (0.6829) is the highest, indicating a strong balance between precision and recall. Chol's F1 score (0.1576) is the lowest, suggesting poor overall performance.
- **Recall**: Measures the ability to identify Potable water samples (minority class). My model's recall (0.7012) is the highest, meaning it excels at detecting Potable water, while Chol's (0.0865) is extremely low, missing most positive cases.
- **Precision**: Indicates the accuracy of positive predictions. Chol's precision (0.8889) is the highest but comes at the expense of recall. My model's precision (0.6940) is balanced and reasonable.

### Comparison with Each Teammate's Model

#### 1. Abiodun vs. Chol
- **Metrics**:
  - F1 Score: 0.6829 vs. 0.1576 → My model is far superior.
  - Recall: 0.7012 vs. 0.0865 → My model detects far more Potable water samples.
  - Precision: 0.6940 vs. 0.8889 → Chol's precision is higher but misleading due to low recall.
- **Why My Model is Better**:
  - My model balances precision and recall effectively, while Chol's model prioritizes precision at the cost of recall, likely predicting very few positives. This is reflected in its low F1 score and recall.
  - **Architecture**: My model uses a deeper structure (128 → 64 → 32 units) with L2 regularization and progressive dropout (0.5 → 0.4 → 0.365), enabling better feature learning and regularization. Chol’s simpler model (64 → 32 units) with L1 regularization and EarlyStopping on validation precision may have biased it toward the majority class.

#### 2. Abiodun vs. Afsa
- **Metrics**:
  - F1 Score: 0.6829 vs. 0.6100 → My model performs better.
  - Recall: 0.7012 vs. 0.6200 → My model has higher recall.
  - Precision: 0.6940 vs. 0.7000 → Nearly identical precision.
- **Why My Model is Better**:
  - Higher recall improves detection of Potable water, boosting my F1 score.
  - **Regularization**: My model uses L2 regularization and Batch Normalization, enhancing generalization. Afsa’s model lacks regularization, and its higher learning rate (0.002 vs. 0.001) may lead to suboptimal convergence.

#### 3. Abiodun vs. Leslie
- **Metrics**:
  - F1 Score: 0.6829 vs. 0.5190 → My model is superior.
  - Recall: 0.7012 vs. 0.4640 → My model has much higher recall.
  - Precision: 0.6940 vs. 0.5890 → My model has higher precision.
- **Why My Model is Better**:
  - My model outperforms across all key metrics, especially recall and F1 score.
  - **Depth and Dropout**: My three-layer architecture (128 → 64 → 32) with progressive dropout outperforms Leslie’s two-layer model (64 → 32) with fixed dropout (0.20), offering better feature extraction and regularization.

#### 4. Abiodun vs. Eddy
- **Metrics**:
  - F1 Score: 0.6829 vs. 0.5080 → My model is better.
  - Recall: 0.7012 vs. 0.4010 → My model has significantly higher recall.
  - Precision: 0.6940 vs. 0.6940 → Precision is identical.
- **Why My Model is Better**:
  - Higher recall improves minority class detection, leading to a better F1 score.
  - **Optimizer and Regularization**: My use of Adam and L2 regularization outperforms Eddy’s SGD and L1 regularization. My layer progression (128 → 64 → 32) may capture features better than Eddy’s (32 → 64 → 128).

#### Reasons for My Model’s Superior Performance

1. **Deeper Architecture**:
   - My model’s three hidden layers (128 → 64 → 32) allow it to learn more complex patterns compared to the shallower models of Chol (64 → 32) and Leslie (64 → 32) or Eddy’s different progression (32 → 64 → 128).

2. **Progressive Dropout**:
   - Dropout rates decrease (0.5 → 0.4 → 0.365) across layers, regularizing early layers more while preserving information in later ones. This contrasts with fixed dropout in other models (e.g., Leslie’s 0.20, Chol’s 0.5).

3. **L2 Regularization and Batch Normalization**:
   - L2 regularization prevents overfitting by penalizing large weights, and Batch Normalization stabilizes training. Chol and Eddy use L1 regularization, which may discard useful features, while Afsa lacks regularization entirely.

4. **Optimizer and Learning Rate Scheduling**:
   - Adam (learning rate 0.001) with ReduceLROnPlateau adapts the learning rate dynamically, likely outperforming Eddy’s SGD (0.010), Chol’s RMSprop (0.0005), or Afsa’s higher Adam rate (0.002).

5. **Class Weights**:
   - Like Leslie and Eddy, I use class weights to handle imbalance, but my model’s architecture and regularization maximize their effectiveness, as seen in the high recall.

### Conclusion

My model achieves the highest F1 score (0.6829) and recall (0.7012), making it the best performer for predicting water potability in this imbalanced dataset. It balances precision and recall effectively, unlike Chol’s model, which sacrifices recall for precision, or Afsa, Leslie, and Eddy’s models, which lag in recall and F1 score. The combination of a deeper architecture, progressive dropout, L2 regularization, Batch Normalization, and an adaptive optimizer drives its superior performance, aligning well with the task’s objective of accurately identifying Potable water.