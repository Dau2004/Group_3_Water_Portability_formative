# Water Quality Model Analysis Report


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
❗ 58% of safe water sources undetected (low Potable recall)
❗ 13% of contaminated water misclassified as safe (Non-Potable precision)

In summary, this experiments underscored the importance of regularization and imbalance handling in neural network design. In future iterations, I would explore alternative architectures like, wider networks(256-unit first layer, adding attention mechanism before final Dense layer), applying quantile transformation to other noisy features, increase portable class weight penalty by 1.5x, use a L1/L2 combo (ElasticNet) in first layer to address these challenges and potentially elevate performance.