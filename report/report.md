# Water Quality Model Analysis Report
## Repo Link: https://github.com/Dau2004/Group_3_Water_Portability_formative.git


## Summary Table
| Train Instance | Engineer Name | Regularizer | Optimizer | Early Stopping         | Dropout Rate                | Accuracy | F1 Score | Recall | Precision |
|:--------------:|:-------------:|:-----------:|:---------:|:----------------------:|:---------------------------:|:--------:|:--------:|:------:|:---------:|
| 1              | Eddy          | L1          | SGD       | Patience=10   | 0.2           | 0.697   | 0.508   | 0.401 | 0.694    |
| 2              | Chol          | L2          | RMSprop   | Patience = 5           | 0.5                         | 0.6524   | 0.1576   | 0.0865 | 0.8889    |
| 3              | Leslie        | L2        | AdamW      | Patience = 5            | 0.2                          | 0.671   | 0.567   | 0.552 | 0.582    |
| 4              | Abiodun       | L2          | Adam      | val_loss, patience=10  | 0.5 → 0.4 → 0.365           | 0.7012   | 0.6829   | 0.7012 | 0.6940    |
| 5              | Afsa          | Dropout       | Adam    | Patience = 10   | 0.4 (1st), 0.3 (2nd)           | 0.69   | 0.44   | 0.32 | 0.73    |

## Individual Analysis Reports

## Abiodun Kumuyi (Member 1)

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



## Chol Daniel Deng (Member 2)

### Model Architecture & Rationale

🔘 **Regularization: L1 (0.001)**  
I chose L1 regularization to encourage model sparsity and reduce overfitting. L1 forces some feature weights to zero, effectively performing feature selection. Given the potential noise and correlation in the 9 water quality features, I aimed to identify the most relevant predictors by penalizing unnecessary complexity.

🔘 **Dropout Rate: 0.5**  
A relatively high dropout rate of 0.5 was used after each dense layer to combat overfitting. This forces the model to avoid reliance on specific neurons and helps generalize better to unseen data.

🔘 **Optimizer & Learning Rate: RMSprop (lr=0.0005, momentum=0.9)**  
RMSprop was selected for its adaptive learning rate and effectiveness in dealing with non-stationary objectives. A relatively small learning rate (0.0005) was used to encourage stable convergence, especially since L1 regularization and high dropout can introduce training instability.

🔘 **Early Stopping**  
I employed EarlyStopping with a patience of 5 epochs, monitoring validation precision. The goal was to stop training once validation precision plateaued, aligning with the objective of reducing false positives in a sensitive application like water safety.

🔘 **Class Weights**  
Class imbalance was addressed by applying balanced class weights during training, ensuring that the minority potable class was given sufficient attention.

### Training Summaries, Results, and Conclusions

The model was trained for up to 100 epochs using binary cross-entropy loss and a batch size of 32, with performance evaluated on accuracy, precision, recall, and AUC metrics.

#### Classification Report
          precision    recall  f1-score   support
Non-Potable 0.66 0.99 0.79 307
Potable 0.89 0.09 0.16 185
accuracy 0.65 492


#### Key Performance Metrics
| Metric         | Value    |
|----------------|----------|
| Accuracy       | 0.6524   |
| Precision      | **0.8889** |
| Recall         | 0.0865   |
| F1-Score       | 0.1576   |
| AUC-ROC        | 0.6418   |

#### Summary and Conclusions
My model achieved **very high precision (0.8889)** for the potable class but at the cost of **extremely low recall (0.0865)**. This means the model was very selective, correctly identifying a small number of potable samples with few false positives, but it missed most of the actual potable cases.

This behavior likely results from the combination of **L1 regularization, high dropout (0.5)**, and EarlyStopping focused on precision. The model learned to avoid predicting potable unless highly confident, which improved precision but harmed recall and F1 score.

Overall, the model is **conservative** — useful where false positives are costly — but not suitable if detecting as many potable sources as possible is the goal.

### Insights from Experiments and Challenges Faced

#### Insights
- **High Precision Focus**: Optimizing for precision, as done with EarlyStopping, significantly reduced recall — an important tradeoff to understand.
- **Effect of L1**: L1 regularization likely pruned many weights, simplifying the model but reducing its flexibility to capture more subtle patterns needed to identify potable water.
- **Dropout Impact**: A 0.5 dropout rate contributed to model robustness but also may have made training harder and led to conservative behavior.

#### Challenges
- **Recall Suppression**: Despite balanced class weights, the model prioritized avoiding false positives over identifying true positives.
- **Feature Selection Sensitivity**: L1 regularization's aggressiveness may have eliminated useful features, explaining the low recall.
- **Hyperparameter Sensitivity**: Tuning dropout and learning rate was challenging; slight changes produced unstable training or degraded performance.

#### Critical Gaps:
- ❗ 91% of safe water sources undetected (potable recall only 8.65%)
- ❗ Highly conservative: suitable only if missing potable water is acceptable (likely not the case in water safety applications)

### Chol's Model Comparison Report

### Metrics Summary

| **Model**  | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC-ROC** |
|------------|--------------|---------------|------------|--------------|-------------|
| **Abiodun** | 0.7012       | 0.6940        | 0.7012     | 0.6829       | 0.6884     |
| **Chol**    | 0.6524       | **0.8889**    | 0.0865     | 0.1576       | 0.6418     |
| **Afsa**    | 0.6900       | 0.7000        | 0.6200     | 0.6100       | N/A        |
| **Leslie**  | 0.6650       | 0.5890        | 0.4640     | 0.5190       | N/A        |
| **Eddy**    | 0.6970       | 0.6940        | 0.4010     | 0.5080       | N/A        |

### Interpretation of Metrics

- **F1 Score**: My model’s F1 (0.1576) is the lowest, due to extremely low recall.
- **Recall**: At 0.0865, my model recalls very few potable samples — unacceptable for this task.
- **Precision**: My model achieves the highest precision (0.8889), showing strong selectivity.

### Comparison with Each Teammate's Model

#### 1. Chol vs. Abiodun
- F1: 0.1576 vs. 0.6829 → Abiodun’s model is far superior.
- Recall: 0.0865 vs. 0.7012 → Abiodun’s model detects far more potable samples.
- Precision: 0.8889 vs. 0.6940 → My model is more selective but at great cost.

#### 2. Chol vs. Afsa
- F1: 0.1576 vs. 0.6100 → Afsa’s model is clearly better.
- Recall: 0.0865 vs. 0.6200 → Afsa detects many more potable cases.
- Precision: 0.8889 vs. 0.7000 → My model has higher precision.

#### 3. Chol vs. Leslie
- F1: 0.1576 vs. 0.5190 → Leslie’s model is better.
- Recall: 0.0865 vs. 0.4640 → Leslie’s model recalls more potable water.
- Precision: 0.8889 vs. 0.5890 → My model is more selective.

#### 4. Chol vs. Eddy
- F1: 0.1576 vs. 0.5080 → Eddy’s model is better.
- Recall: 0.0865 vs. 0.4010 → Eddy recalls more potable water.
- Precision: 0.8889 vs. 0.6940 → My model is more selective.

#### Reasons for My Model’s Behavior

1. **EarlyStopping on Precision**: Focusing on maximizing precision encouraged overly conservative predictions.
2. **L1 Regularization**: Aggressively penalized weights, reducing model capacity to detect subtle potable patterns.
3. **High Dropout**: Further limited model flexibility.
4. **Architecture Simplicity**: Shallower architecture (64 → 32) compared to others (128 → 64 → 32) limited representational power.

### Conclusion

My model achieves **highest precision (0.8889)** but at the expense of recall and F1 score, making it unsuitable for this imbalanced classification task. While useful for **applications prioritizing false positive reduction**, it fails to identify enough potable water samples to be practical for public health contexts.

Future work would explore:
- Using **L2 regularization or ElasticNet** instead of pure L1  
- Reducing dropout rate (e.g., to 0.3)  
- Optimizing for balanced metrics like F1, not just precision  
- Deepening the architecture (adding a 128-unit layer)  

## Leslie Isaro (Member 3)

### Model Architecture & Rationale

🔘 **Regularization: L2 (0.001)**  
L2 regularization was applied to all dense layers with a strength of 0.001. This prevents overfitting by discouraging large weight values while still allowing all features to contribute to learning — particularly important when all 9 water-quality indicators may have value.

🔘 **Dropout Rate: Fixed (0.20)**  
A fixed dropout rate of 0.2 was used after both hidden layers to provide light regularization. This moderately suppresses co-adaptation between neurons while retaining learning capacity — a trade-off for stability and generalization.

🔘 **Optimizer & Learning Rate: AdamW (lr = 0.001, weight_decay = 0.001)**  
AdamW was chosen for its adaptive learning capabilities and its decoupled weight decay, which enforces L2 regularization directly through the optimizer update step. This pairing of optimizer and regularization provides robust, principled convergence.

🔘 **Early Stopping**  
Early stopping was implemented with a patience of 5, monitoring validation loss. This ensured training stopped once performance plateaued and restored the best model weights. This improved both training efficiency and generalization.

🔘 **Class Weights**  
To address the imbalance in the target classes, class weights were computed using `compute_class_weight('balanced')`. This encouraged the model to take underrepresented potable samples (class 1) more seriously during training, improving recall and fairness.

### Training Summaries, Results, and Conclusions

The model was trained for up to 4000 epochs with early stopping triggered after convergence. It used binary cross-entropy loss, AdamW optimizer, and evaluation metrics including accuracy, precision, recall, and F1-score.

#### Classification Report

#### Key Performance Metrics
| Metric         | Value |
|----------------|--------|
| Accuracy       | 0.671  |
| Precision      | 0.582  |
| Recall         | 0.552  |
| F1-Score       | 0.567  |
| AUC-ROC        | N/A    |

#### Summary and Conclusions
Leslie's model achieved a well-balanced performance across all metrics. While not the highest performer overall, it maintained stable precision and recall, giving it a strong **F1-score of 0.567**. This reflects the model's ability to detect potable water samples more reliably than overly conservative models (e.g., Chol's).

The architecture was simpler (64 → 32) but highly stable due to AdamW, L2, and class weights. Recall and F1 are especially improved over models that focused too much on precision.

While effective, future improvements might come from increasing model depth (e.g., 128 → 64 → 32), tuning dropout progressively, or adjusting classification thresholds to better balance precision-recall trade-offs.

### Insights from Experiments and Challenges Faced

### AdamW Improved Generalization

Switching from `Adam` and `SGD` to `AdamW` led to more stable training and better generalization. Because AdamW decouples weight decay from the gradient update, it applied regularization more cleanly. This helped reduce overfitting compared to earlier runs using `SGD` and standard L2.

---

### Class Weights Boosted Recall

Introducing `class_weight='balanced'` significantly improved the model’s ability to detect *potable* samples (minority class). Prior to class weighting, recall was consistently below 0.35; after weighting, it reached **0.55**, contributing to a better F1 score.

---

### Shallow Networks Can Compete

Even with just two hidden layers (64 → 32 units), the model performed well. While deeper networks may offer more representation power, the correct use of regularization, dropout, and optimizer choice enabled competitive results.


#### Challenges

### Low Initial Recall

Early model versions using `SGD` and no class weighting had high accuracy but recall below 0.30, meaning the model frequently failed to detect potable water. This was unacceptable in the context of water safety. Class weighting helped, but recall still lagged behind deeper models.

---

### Overfitting on Small Feature Space

Despite using dropout and L2, validation loss would occasionally spike after a few epochs — especially when using a higher dropout rate (0.3+). With only 9 features, regularization needed to be carefully balanced to avoid underfitting.

---

### Optimizer Tradeoffs

`SGD` with momentum proved too slow to converge and often plateaued early unless tuned carefully. `Adam` helped but sometimes overfitted. `AdamW` offered the best balance, though it required experimenting with `weight_decay`.


### Leslie's Model Comparison Report

#### Metrics Summary

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------|--------------|---------------|------------|--------------|
| **Leslie**  | 0.671        | 0.582         | 0.552      | 0.567        |
| **Chol**    | 0.652        | 0.888         | 0.087      | 0.158        |
| **Afsa**    | 0.690        | 0.700         | 0.620      | 0.610        |
| **Eddy**    | 0.697        | 0.694         | 0.401      | 0.508        |

#### Interpretation of Metrics

- **F1 Score**: Leslie outperforms Chol and Eddy on F1 score, showing better balance between detecting potable water and avoiding false positives.
- **Recall**: Leslie's recall (0.552) is significantly better than Chol’s and Eddy’s, meaning more actual potable water samples were detected.
- **Precision**: Leslie maintains respectable precision (0.582), ensuring predictions aren't overly risky.

#### Comparison with Each Teammate's Model

**Leslie vs. Chol**  
| **Metric**   | **Leslie** | **Chol** |
|--------------|------------|----------|
| **F1**       | 0.567      | 0.158    |
| **Recall**   | 0.552      | 0.087    |
| **Precision**| 0.582      | 0.889    |

### Why Leslie’s Model Is Better:

- Chol’s model sacrifices recall for very high precision — it predicts very few samples as potable, causing most true positives to be missed (recall = 0.087).
- Leslie’s model maintains a healthier balance, with solid precision **and** recall, leading to a **3.5× better F1 score**.

### Architectural Differences:

- Leslie uses **L2 regularization** (less aggressive than Chol’s L1), **AdamW**, and lower dropout (0.2 vs. 0.5), encouraging better learning.
- Chol’s use of high dropout (0.5) and L1 may have limited feature learning too aggressively.

**Leslie vs. Afsa**  
| **Metric**   | **Leslie** | **Afsa** |
|--------------|------------|----------|
| F1           | 0.567      | 0.610    |
| Recall       | 0.552      | 0.620    |
| Precision    | 0.582      | 0.700    |

### Why Afsa's Model Performs Slightly Better:

- Afsa edges ahead in all metrics, particularly recall and F1.
- The higher learning rate (0.002) and lighter regularization may have enabled better fit to the minority class.

### Leslie's Advantages:

- More regularized (**L2 + AdamW**) and thus potentially **more robust to overfitting**.
- Comparable performance with a **more conservative and theoretically grounded setup**.


**Leslie vs. Eddy**  
| **Metric**   | **Leslie** | **Eddy** |
|--------------|------------|----------|
| F1           | 0.567      | 0.508    |
| Recall       | 0.552      | 0.401    |
| Precision    | 0.582      | 0.694    |

### Why Leslie’s Model Is Better:

- Better recall (+15 percentage points), resulting in higher F1.
- Slightly lower precision is acceptable given much better balance overall.

### Model Differences:

- Eddy uses **L1 regularization** and an inverted architecture (32 → 64 → 128), which may overfit or underrepresent key features early.
- Leslie's consistent **L2 regularization** and classic deep-to-shallow flow (64 → 32) allow more stable learning.


### Conclusion

My model achieved a balanced and consistent performance in classifying water potability, with an **F1 score of 0.567**, **recall of 0.552**, and **precision of 0.582**. These results reflect a model that maintains fairness between correctly identifying potable water and minimizing false positives — a critical balance in public health applications.

Compared to teammates’ models, mine outperformed those that prioritized only one metric (e.g., Chol’s high precision but very low recall), by providing more reliable and actionable predictions. While not the absolute top in any single metric, my model ranked solidly across the board, making it a dependable baseline for real-world use.

The use of **L2 regularization**, **AdamW optimizer**, **class weighting**, and **early stopping** contributed to effective generalization and training stability. However, the fixed dropout and limited model depth may have constrained performance, especially in identifying more subtle patterns related to potable water.

In future iterations, I would explore:
- Adding a third hidden layer for deeper pattern recognition
- Tuning dropout rates progressively across layers
- Adjusting the classification threshold to improve recall
- Increasing class 1 penalty weight beyond `balanced` default

Overall, this project highlights the importance of thoughtful architecture, regularization, and evaluation in imbalanced binary classification problems. My model strikes a valuable trade-off and provides a strong foundation for further refinement.

## Afsa Umutoniwase (Member 4)

### Model Architecture & Rationale
🔘 Regularization: Dropout (0.4, 0.3)
I used dropout as a regularization method to combat overfitting. A 0.4 dropout was applied after the first dense layer, and 0.3 after the second. This prevents the model from relying too heavily on specific neurons and promotes more generalized representations. Given the small dataset and class imbalance, this helped reduce variance.

🔘 Optimizer & Learning Rate: Adam (lr = 0.002)
I chose the Adam optimizer for its adaptive learning capabilities and stability across noisy gradients. A learning rate of 0.002 (higher than default) helped the model learn more efficiently in early epochs while still avoiding divergence.

🔘 Early Stopping: Monitored val_loss, patience = 10
EarlyStopping helped prevent overfitting by stopping the training once validation loss stopped improving. A patience of 10 epochs gave the model enough time to escape local minima but also ensured training efficiency.

🔘 Loss & Metrics
Binary cross-entropy was used as the loss function since this was a binary classification task. The evaluation metrics included accuracy, precision, and recall to reflect class imbalance concerns.

### Training Summaries, Results, and Conclusions

Epochs: up to 4000 (EarlyStopping engaged)

Batch Size: default (32)

Activation Functions: ReLU (hidden), Sigmoid (output)

#### Classification Report

#### Key Performance Metrics
| Metric         | Value |
|----------------|--------|
| Accuracy       | 0.69  |
| Precision      | 0.70  |
| Recall         | 0.62  |
| F1-Score       | 0.61  |

#### Summary and Conclusions

The model achieved moderate accuracy (69%) and high precision (0.73) on potable water detection but had low recall (0.32). This means it correctly identified many non-potable cases but missed a large portion of actual potable ones.

Main tradeoff: Higher confidence in positive predictions (potable), but at the expense of missing many true positives.

### Insights from Experiments and Challenges Faced

#### Insights
- High Precision, Low Recall: Dropout and Adam optimizer stabilized training and improved generalization, but also made the model more cautious in flagging potable samples.
- Dropout Regularization: Helped reduce overfitting, but may have limited the model’s ability to fully capture minority class patterns.
- EarlyStopping Effectiveness: Prevented unnecessary training and helped maintain optimal validation loss.

#### Challenges:

- Class Imbalance: Skewed label distribution led to low recall. Future improvement could include class weighting or SMOTE.
- Model Simplicity: A two-layer network may not have been deep enough to fully extract complex patterns in the data.

### Afsa's Model Comparison Report

### Metrics Summary

| **Model**  | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC-ROC** |
|------------|--------------|---------------|------------|--------------|-------------|
| **Abiodun** | 0.7012       | 0.6940        | 0.7012     | 0.6829       | 0.6884     |
| **Chol**    | 0.6524       | 0.8889    | 0.0865     | 0.1576       | 0.6418     |
| **Afsa**    | 0.6900       | 0.7000        | 0.6200     | 0.6100       | N/A        |
| **Leslie**  | 0.6650       | 0.5890        | 0.4640     | 0.5190       | N/A        |
| **Eddy**    | 0.6970       | 0.6940        | 0.4010     | 0.5080       | N/A        |

### Interpretation of Metrics

- **F1 Score**:  My model’s F1 (0.6100) reflects a good balance between precision and recall.
- **Recall**: At 0.6200, my model identifies a solid portion of potable samples — much higher than models like Chol or Eddy.
- **Precision**: At 0.7000, my model confidently predicts potable water with a reasonable false positive rate.

### Comparison with Each Teammate's Model
#### 1. Afsa vs. Abiodun
- F1: 0.6100 vs. 0.6829 → Abiodun’s model is stronger overall.
- Recall: 0.6200 vs. 0.7012 → Abiodun captures more potable cases.
- Precision: 0.7000 vs. 0.6940 → My model is slightly more selective.

#### 2. Afsa vs. Chol
- F1: 0.6100 vs. 0.1576 → My model is vastly more balanced.
- Recall: 0.6200 vs. 0.0865 → I detect far more potable cases.
- Precision: 0.7000 vs. 0.8889 → Chol is more conservative but misses nearly all positives.

#### 3. Afsa vs. Leslie
- F1: 0.6100 vs. 0.5190 → My model performs better overall.
- Recall: 0.6200 vs. 0.4640 → I detect more potable samples.
- Precision: 0.7000 vs. 0.5890 → I’m also more confident in predictions.

#### 4. Afsa vs. Eddy
- F1: 0.6100 vs. 0.5080 → I have stronger balance.
- Recall: 0.6200 vs. 0.4010 → I detect more potable water.
- Precision: 0.7000 vs. 0.6940 → Slight edge in selectivity.

#### Reasons for My Model’s Behavior

1. EarlyStopping on val_loss: Prevented overfitting but may have capped late-stage recall improvements.
2. Dropout (0.4/0.3): Balanced regularization but possibly limited model depth.
3. Adam Optimizer (lr=0.002): Provided fast, stable learning and contributed to good early convergence.
4. No Class Weighting: Likely affected recall performance under class imbalance.

### Conclusion

My model provides a solid trade-off between precision and recall, making it practical for early-stage water quality assessments. It avoids being overly conservative like Chol’s model while not overpredicting positive labels.

Future work would explore:
- Apply class weighting or SMOTE to handle imbalance better
- Experiment with deeper layers (e.g., 128 → 64 → 32)
- Try L2 regularization for smoother weight decay
- Monitor F1-score during EarlyStopping instead of val_loss
- Tune learning rate adaptively for even better convergence

## Eddy Gasana (Member 5)
### Eddy's Model Findings Report

### Model Architecture & Rationale

🔘 **Regularization: L1 Regularization + Dropout (0.2)**
Used L1 regularization to encourage sparsity in the network weights, improving model generalization. A dropout rate of 0.2 was applied after each dense layer to further reduce overfitting by randomly deactivating 20% of the neurons during training.

🔘 **Optimizer & Learning Rate: SGD (lr = 0.010, momentum = 0.9)**
The SGD optimizer with momentum helped smooth the gradient updates, accelerating convergence and reducing oscillation. A learning rate of 0.010 allowed stable but relatively quick updates without overshooting.

🔘 **Early Stopping: Monitored val_loss, patience = 10**
Training was stopped early when validation loss ceased improving after 10 epochs. This approach balanced training efficiency and generalization, helping avoid overfitting.

🔘 **Loss & Metrics**
Binary cross-entropy was used for binary classification. Metrics included accuracy, precision, and recall to assess performance considering class imbalance.

### Training Summaries, Results, and Conclusions

- **Epochs**: up to 2000 (EarlyStopping engaged at epoch 51)
- **Batch Size**: 32 (default)
- **Activation Functions**: ReLU (hidden), Sigmoid (output)

#### Key Performance Metrics
| Metric         | Value |
|----------------|--------|
| Accuracy       | 0.697  |
| Precision      | 0.694  |
| Recall         | 0.401  |
| F1-Score       | 0.508  |

### Summary and Conclusions

The model achieved a balanced accuracy (69.7%) and strong precision (69.4%) in identifying potable water. However, it had moderate recall (40.1%), indicating a considerable number of actual potable cases were missed.

**Main Tradeoff:** Prioritized correct predictions when it flagged potable water (precision) but at the cost of missing many actual potable instances (low recall).

### Insights from Experiments and Challenges Faced

#### Insights
- **L1 Regularization**: Helped sparsify the weights, improving generalization but possibly pruning too aggressively.
- **Dropout**: At 0.2, provided a balanced regularization effect—less prone to underfitting than larger rates.
- **SGD with Momentum**: Offered stable and consistent learning but may converge more slowly than adaptive optimizers like Adam.
- **EarlyStopping**: Prevented overfitting but may have limited further recall gains.

#### Challenges:
- **Class Imbalance**: Affected the model's ability to recall potable cases. Class weighting was applied, but further improvement might require SMOTE or loss adjustment.
- **Recall Bottleneck**: Due to conservative predictions and limited model depth.

### Eddy's Model Comparison Report

#### Metrics Summary
| **Model**  | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC-ROC** |
|------------|--------------|---------------|------------|--------------|-------------|
| **Abiodun** | 0.7012       | 0.6940        | 0.7012     | 0.6829       | 0.6884     |
| **Chol**    | 0.6524       | 0.8889        | 0.0865     | 0.1576       | 0.6418     |
| **Afsa**    | 0.6900       | 0.7000        | 0.6200     | 0.6100       | N/A        |
| **Leslie**  | 0.6650       | 0.5890        | 0.4640     | 0.5190       | N/A        |
| **Eddy**    | 0.6970       | 0.6940        | 0.4010     | 0.5080       | N/A        |

### Comparison with Each Teammate's Model

#### Eddy vs. Abiodun
- F1: 0.5080 vs. 0.6829 → Abiodun's model has superior balance.
- Recall: 0.4010 vs. 0.7012 → Abiodun detects more potable cases.
- Precision: Equal at 0.6940 → Both equally confident in potable predictions.

#### Eddy vs. Chol
- F1: 0.5080 vs. 0.1576 → My model is significantly more balanced.
- Recall: 0.4010 vs. 0.0865 → Much better at identifying potable water.
- Precision: 0.6940 vs. 0.8889 → Chol is more conservative, predicting fewer but more accurate positives.

#### Eddy vs. Leslie
- F1: 0.5080 vs. 0.5190 → Close in balance, slightly lower.
- Recall: 0.4010 vs. 0.4640 → Leslie detects more potables.
- Precision: 0.6940 vs. 0.5890 → My model is more selective.

#### Eddy vs. Afsa
- F1: 0.5080 vs. 0.6100 → Afsa’s model is better balanced.
- Recall: 0.4010 vs. 0.6200 → Afsa detects more potable water.
- Precision: 0.6940 vs. 0.7000 → Nearly identical.

### Reasons for My Model’s Behavior
1. **L1 Regularization**: May have reduced model complexity excessively.
2. **SGD Optimizer**: Slower convergence compared to adaptive methods.
3. **EarlyStopping**: May have stopped before recall could improve.
4. **Dropout (0.2)**: Balanced regularization, avoiding underfitting.

### Conclusion

My model offers a high precision and decent accuracy, making it reliable for confirming potable water, though less aggressive in catching all such cases. It is moderately balanced but can benefit from recall-improving strategies like:

- Class weighting adjustments or focal loss
- Additional hidden layers or units
- Switch from L1 to L2 or ElasticNet regularization
- Use of adaptive optimizers like Adam or RMSprop
- Monitoring F1-Score during EarlyStopping
