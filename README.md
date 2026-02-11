BPM Prediction – Weak Signal Modeling & Progressive Capacity Scaling

Project Overview

This project was developed in the context of a Kaggle-style regression challenge.

Objective: Predict BeatsPerMinute (BPM) from structured audio-derived features.

The central difficulty of this task was the extremely weak signal-to-noise ratio.
Feature-target correlations were low, linear structure was limited, and much of the predictive information was embedded in non-linear interactions.

This project therefore focuses on progressive model capacity scaling, systematic experimentation, and disciplined validation under noisy conditions.

Core Challenge

The dataset exhibited:

- Low marginal feature correlation
- High noise component
- Interaction-driven predictive structure
- Risk of overfitting when increasing model capacity

The key question was not simply “which model works best?”
But rather:

How do we extract stable signal from weakly informative tabular data without exploding variance?

Experimental Strategy

A structured escalation approach was adopted.

Each modeling phase aimed to test a specific hypothesis about the data.

Phase 1 – Linear Signal Detection

Goal: Detect linear structure and estimate irreducible error.

Models:
- OLS
- Polynomial terms
- Interaction terms
- Statistical significance filtering

Conclusion:
Linear capacity was insufficient → strong underfitting confirmed.

Phase 2 – Variance Control via Regularization

Goal: Test sparsity assumptions and reduce overfitting risk.

Models:
- Ridge
- Lasso
- Hyperparameter search on α
- Cluster-wise modeling

Insight:
Regularization improved stability but did not meaningfully increase predictive power, reinforcing the hypothesis of predominantly non-linear signal.

Phase 3 – Latent Structure Exploration

Goal: Identify hidden structure in feature space.

Techniques:

- PCA (variance concentration analysis)
- K-Means
- Gaussian Mixture Models

Cluster assignments were integrated as additional features.

Finding:
Some sub-populations existed, but cluster separability remained weak — consistent with noisy signal distribution.

Phase 4 – Non-Linear Ensemble Modeling

Goal: Capture high-order interactions while controlling variance.

Models:
- Random Forest (cross-validated)
- Gradient Boosting
- HistGradientBoostingRegressor

Tree ensembles significantly outperformed linear models, confirming the importance of hierarchical feature interactions.

Careful tuning of:

- Learning rate
- Depth
- Regularization parameters
- Early stopping
was critical to avoid overfitting.

Phase 5 – High-Capacity Neural Networks

Goal: Test whether additional representational capacity could extract residual signal.

Architectures tested:
- Shallow MLP
- Regularized deep network (BatchNorm + Dropout)
- Symmetric bottleneck architecture (256-128-64-128-256)

Controls:
- Early stopping
- Learning rate scheduling
- L2 regularization
- Validation monitoring

Result:
Neural networks achieved competitive performance but were highly sensitive to regularization due to the weak signal regime.

Feature Engineering as Signal Amplification

To increase effective signal strength, an extensive feature expansion pipeline was built:
- Log / sqrt transforms
- Pairwise interaction terms
- Ratio features
- Relative normalization
- Quantile binning
- Ranking transformations

This increased hypothesis space while relying on strong regularization downstream.

Final Model Selection

Final model: HistGradientBoostingRegressor

Reasons:
- Strong performance on structured tabular data
- Built-in regularization
- Computational efficiency
- Stable bias-variance tradeoff under noisy regimes

Model selection was based on:
- Validation RMSE
- Cross-validation stability
- Sensitivity to hyperparameters
- Generalization consistency

Validation Rigor
- Strict train/validation separation
- No feature leakage
- Cross-validation for ensemble models
- Consistent preprocessing pipeline between train and test
- Reproducibility via fixed random states

What This Project Demonstrates
- Ability to operate in weak-signal environments
- Systematic experimental design
- Progressive model capacity reasoning
- Bias-variance tradeoff management
- High-dimensional feature engineering
- Ensemble optimization
- Neural network regularization under noise
- Research-style hypothesis testing

This project is less about achieving a single leaderboard score and more about demonstrating disciplined ML reasoning under uncertainty.

Technical Stack
- Python
- NumPy / Pandas
- Scikit-learn
- Statsmodels
- TensorFlow / Keras
- Matplotlib / Seaborn

Author
Alexandre Le Droucpeet
Ajouter une section “Failure Analysis & Model Diagnostics”

Ou reformuler tout ça en version encore plus minimaliste et élite (style Google Research / DeepMind tone)
