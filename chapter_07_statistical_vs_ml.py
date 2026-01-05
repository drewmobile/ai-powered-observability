"""
Chapter 7: Statistical vs. ML-Based Approaches
================================================

Code examples demonstrating the spectrum of anomaly detection methods:

Statistical Methods:
- Z-score and standard deviation
- Percentile-based methods  
- Interquartile Range (IQR)
- Moving averages (SMA, EMA)
- Seasonal decomposition (STL)

Machine Learning Methods:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- DBSCAN-based detection
- Autoencoder (neural network)

Hybrid Approaches:
- Statistical screening + ML refinement
- Ensemble detection
- Tiered detection systems

Implementation Considerations:
- Threshold setting strategies
- Seasonality handling
- Cold start solutions
- Feedback integration
- Evaluation metrics

Prerequisites:
    pip install numpy scikit-learn scipy

This chapter focuses on algorithm comparison and selection guidance.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
import warnings

# scikit-learn imports
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# scipy for statistical methods
from scipy import stats
from scipy.signal import find_peaks


# =============================================================================
# DATA GENERATION FOR TESTING
# =============================================================================

def generate_test_data(
    n_points: int = 1000,
    anomaly_fraction: float = 0.05,
    include_seasonality: bool = True,
    include_trend: bool = False,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data with known anomalies.
    
    Returns:
        data: The time series values
        timestamps: Corresponding timestamps
        labels: 1 for anomaly, 0 for normal
    """
    np.random.seed(42)
    
    # Base signal
    t = np.arange(n_points)
    
    # Add seasonality (daily pattern over ~100 points)
    if include_seasonality:
        daily_pattern = 10 * np.sin(2 * np.pi * t / 100)
        weekly_pattern = 5 * np.sin(2 * np.pi * t / 700)
        signal = daily_pattern + weekly_pattern
    else:
        signal = np.zeros(n_points)
    
    # Add trend
    if include_trend:
        signal += 0.01 * t
    
    # Add baseline and noise
    signal += 50 + noise_level * 50 * np.random.randn(n_points)
    
    # Inject anomalies
    n_anomalies = int(n_points * anomaly_fraction)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    labels = np.zeros(n_points)
    labels[anomaly_indices] = 1
    
    # Create different types of anomalies
    for i, idx in enumerate(anomaly_indices):
        anomaly_type = i % 4
        if anomaly_type == 0:
            # Point anomaly - high spike
            signal[idx] += 30 + 20 * np.random.rand()
        elif anomaly_type == 1:
            # Point anomaly - low dip
            signal[idx] -= 30 + 20 * np.random.rand()
        elif anomaly_type == 2:
            # Gradual anomaly - small persistent change
            if idx + 5 < n_points:
                signal[idx:idx+5] += 15
        else:
            # Variance anomaly - increased noise
            if idx + 3 < n_points:
                signal[idx:idx+3] += 10 * np.random.randn(min(3, n_points - idx))
    
    # Generate timestamps
    base_time = datetime(2024, 1, 1)
    timestamps = np.array([base_time + timedelta(minutes=5*i) for i in range(n_points)])
    
    return signal, timestamps, labels


# =============================================================================
# SECTION 1: STATISTICAL ANOMALY DETECTION METHODS
# =============================================================================

class AnomalyDetector(ABC):
    """Abstract base class for all anomaly detectors."""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'AnomalyDetector':
        """Fit the detector to training data."""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1=anomaly, 0=normal)."""
        pass
    
    @abstractmethod
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        pass


class ZScoreDetector(AnomalyDetector):
    """
    Z-Score based anomaly detection.
    
    The most basic statistical approach:
    z = (x - μ) / σ
    
    Points with |z| > threshold are flagged as anomalies.
    
    Best for:
    - Normally distributed data
    - Stable mean and variance
    - Point anomalies
    
    Limitations:
    - Assumes normality
    - Sensitive to outliers in training data
    - No temporal awareness
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        Args:
            threshold: Number of standard deviations for anomaly detection.
                      3.0 = 99.7% confidence for normal distribution
        """
        self.threshold = threshold
        self.mean = None
        self.std = None
    
    def fit(self, data: np.ndarray) -> 'ZScoreDetector':
        """Compute mean and standard deviation from training data."""
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std == 0:
            self.std = 1e-10  # Avoid division by zero
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Calculate z-scores (absolute value)."""
        return np.abs((data - self.mean) / self.std)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies based on z-score threshold."""
        scores = self.score(data)
        return (scores > self.threshold).astype(int)
    
    def get_params(self) -> Dict:
        """Return learned parameters."""
        return {
            'mean': self.mean,
            'std': self.std,
            'threshold': self.threshold
        }


class PercentileDetector(AnomalyDetector):
    """
    Percentile-based anomaly detection.
    
    Flags values below lower_percentile or above upper_percentile.
    
    Advantages over Z-score:
    - No normality assumption
    - Uses empirical distribution
    - More robust to skewed data
    
    Best for:
    - Non-normal distributions
    - Heavy-tailed data (like latency)
    - When you want to flag extreme values
    """
    
    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        """
        Args:
            lower_percentile: Values below this percentile are anomalies
            upper_percentile: Values above this percentile are anomalies
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, data: np.ndarray) -> 'PercentileDetector':
        """Compute percentile bounds from training data."""
        self.lower_bound = np.percentile(data, self.lower_percentile)
        self.upper_bound = np.percentile(data, self.upper_percentile)
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Score based on distance from bounds.
        0 = within bounds, positive = outside bounds
        """
        scores = np.zeros_like(data, dtype=float)
        
        # Score for values below lower bound
        below_mask = data < self.lower_bound
        scores[below_mask] = (self.lower_bound - data[below_mask]) / max(abs(self.lower_bound), 1)
        
        # Score for values above upper bound
        above_mask = data > self.upper_bound
        scores[above_mask] = (data[above_mask] - self.upper_bound) / max(abs(self.upper_bound), 1)
        
        return scores
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies based on percentile bounds."""
        return ((data < self.lower_bound) | (data > self.upper_bound)).astype(int)
    
    def get_params(self) -> Dict:
        """Return learned parameters."""
        return {
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'lower_percentile': self.lower_percentile,
            'upper_percentile': self.upper_percentile
        }


class IQRDetector(AnomalyDetector):
    """
    Interquartile Range (IQR) based anomaly detection.
    
    Outliers are defined as values beyond:
    Q1 - k*IQR  or  Q3 + k*IQR
    
    Where k is typically 1.5 (outliers) or 3.0 (extreme outliers).
    
    Advantages:
    - Very robust to outliers in training data
    - No distribution assumptions
    - Simple and interpretable
    
    Best for:
    - Initial outlier screening
    - Data with potential contamination
    - Exploratory analysis
    """
    
    def __init__(self, k: float = 1.5):
        """
        Args:
            k: Multiplier for IQR. 1.5 = standard outliers, 3.0 = extreme
        """
        self.k = k
        self.q1 = None
        self.q3 = None
        self.iqr = None
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, data: np.ndarray) -> 'IQRDetector':
        """Compute IQR bounds from training data."""
        self.q1 = np.percentile(data, 25)
        self.q3 = np.percentile(data, 75)
        self.iqr = self.q3 - self.q1
        self.lower_bound = self.q1 - self.k * self.iqr
        self.upper_bound = self.q3 + self.k * self.iqr
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Score based on distance from IQR bounds."""
        scores = np.zeros_like(data, dtype=float)
        
        below_mask = data < self.lower_bound
        scores[below_mask] = (self.lower_bound - data[below_mask]) / max(self.iqr, 1e-10)
        
        above_mask = data > self.upper_bound
        scores[above_mask] = (data[above_mask] - self.upper_bound) / max(self.iqr, 1e-10)
        
        return scores
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies based on IQR bounds."""
        return ((data < self.lower_bound) | (data > self.upper_bound)).astype(int)


class MovingAverageDetector(AnomalyDetector):
    """
    Moving Average based anomaly detection.
    
    Detects points that deviate significantly from the moving average.
    Supports both Simple Moving Average (SMA) and Exponential Moving Average (EMA).
    
    Advantages:
    - Adapts to trends
    - Temporal awareness
    - Smooth baseline
    
    Best for:
    - Time series with trends
    - Data with gradual drift
    - When recent history matters more
    """
    
    def __init__(
        self,
        window_size: int = 20,
        threshold_std: float = 3.0,
        method: str = 'sma'
    ):
        """
        Args:
            window_size: Size of moving window (for SMA) or effective window (for EMA)
            threshold_std: Standard deviations for anomaly threshold
            method: 'sma' for Simple Moving Average, 'ema' for Exponential
        """
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.method = method
        self.baseline = None
        self.residual_std = None
        
        # EMA smoothing factor (alpha = 2/(window+1) is common)
        self.alpha = 2.0 / (window_size + 1) if method == 'ema' else None
    
    def _compute_sma(self, data: np.ndarray) -> np.ndarray:
        """Compute Simple Moving Average."""
        sma = np.convolve(data, np.ones(self.window_size)/self.window_size, mode='valid')
        # Pad the beginning with the first computed value
        padding = np.full(self.window_size - 1, sma[0])
        return np.concatenate([padding, sma])
    
    def _compute_ema(self, data: np.ndarray) -> np.ndarray:
        """Compute Exponential Moving Average."""
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = self.alpha * data[i] + (1 - self.alpha) * ema[i-1]
        return ema
    
    def fit(self, data: np.ndarray) -> 'MovingAverageDetector':
        """Compute moving average baseline and residual statistics."""
        if self.method == 'sma':
            self.baseline = self._compute_sma(data)
        else:
            self.baseline = self._compute_ema(data)
        
        # Compute residuals and their standard deviation
        residuals = data - self.baseline
        self.residual_std = np.std(residuals)
        
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Score based on deviation from moving average."""
        if self.method == 'sma':
            baseline = self._compute_sma(data)
        else:
            baseline = self._compute_ema(data)
        
        residuals = data - baseline
        return np.abs(residuals) / max(self.residual_std, 1e-10)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies based on deviation threshold."""
        scores = self.score(data)
        return (scores > self.threshold_std).astype(int)


class SeasonalDecompositionDetector(AnomalyDetector):
    """
    Seasonal Decomposition based anomaly detection.
    
    Decomposes time series into:
    y(t) = trend(t) + seasonal(t) + residual(t)
    
    Anomalies are detected in the residual component.
    
    Advantages:
    - Handles seasonal patterns
    - Separates different signal components
    - Reduces false positives from seasonality
    
    Best for:
    - Data with known periodic patterns
    - Daily/weekly cycles
    - When seasonality causes false positives
    """
    
    def __init__(
        self,
        period: int = 24,  # e.g., 24 for hourly data with daily seasonality
        threshold_std: float = 3.0
    ):
        """
        Args:
            period: Length of seasonal cycle
            threshold_std: Standard deviations for anomaly threshold
        """
        self.period = period
        self.threshold_std = threshold_std
        self.seasonal_pattern = None
        self.residual_std = None
    
    def _decompose(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple additive decomposition.
        
        For production, consider using statsmodels.tsa.seasonal.STL
        """
        n = len(data)
        
        # Estimate trend using centered moving average
        if n < self.period * 2:
            trend = np.full(n, np.mean(data))
        else:
            # Centered moving average
            trend = np.convolve(data, np.ones(self.period)/self.period, mode='same')
            # Handle edges
            trend[:self.period//2] = trend[self.period//2]
            trend[-(self.period//2):] = trend[-(self.period//2)-1]
        
        # Detrended series
        detrended = data - trend
        
        # Estimate seasonal component by averaging over periods
        seasonal = np.zeros(n)
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            seasonal_value = np.mean(detrended[indices])
            seasonal[indices] = seasonal_value
        
        # Residual
        residual = data - trend - seasonal
        
        return trend, seasonal, residual
    
    def fit(self, data: np.ndarray) -> 'SeasonalDecompositionDetector':
        """Fit decomposition and learn residual statistics."""
        trend, seasonal, residual = self._decompose(data)
        
        # Store seasonal pattern for one period
        self.seasonal_pattern = seasonal[:self.period]
        self.residual_std = np.std(residual)
        
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Score based on residual magnitude."""
        _, _, residual = self._decompose(data)
        return np.abs(residual) / max(self.residual_std, 1e-10)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies based on residual threshold."""
        scores = self.score(data)
        return (scores > self.threshold_std).astype(int)


# =============================================================================
# SECTION 2: MACHINE LEARNING ANOMALY DETECTION METHODS
# =============================================================================

class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest based anomaly detection.
    
    Key insight: Anomalies are "few and different" - they're easier
    to isolate with random partitioning than normal points.
    
    Advantages:
    - Works well with high-dimensional data
    - No distribution assumptions
    - Handles mixed data types
    - Computationally efficient
    
    Best for:
    - Multivariate data
    - Unknown anomaly types
    - Large datasets
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of isolation trees
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'IsolationForestDetector':
        """Fit isolation forest to training data."""
        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.model.fit(data)
        self.is_fitted = True
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores.
        
        Isolation Forest returns negative scores (more negative = more anomalous).
        We invert to make positive = more anomalous.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # decision_function returns negative scores for anomalies
        return -self.model.decision_function(data)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Isolation Forest returns 1 for normal, -1 for anomaly.
        We convert to 0 for normal, 1 for anomaly.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        predictions = self.model.predict(data)
        return (predictions == -1).astype(int)


class LOFDetector(AnomalyDetector):
    """
    Local Outlier Factor (LOF) based anomaly detection.
    
    Compares local density of a point to its neighbors.
    Points with lower density than neighbors are anomalies.
    
    Advantages:
    - Detects local anomalies
    - Handles varying density regions
    - Good for clustered data
    
    Limitations:
    - Computationally expensive for large datasets
    - Sensitive to n_neighbors parameter
    - Struggles with high dimensions
    
    Best for:
    - Moderate-sized datasets
    - Data with varying densities
    - When local context matters
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05
    ):
        """
        Args:
            n_neighbors: Number of neighbors for density estimation
            contamination: Expected proportion of anomalies
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True  # Enable predict() method
        )
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'LOFDetector':
        """Fit LOF to training data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.model.fit(data)
        self.is_fitted = True
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # Negative LOF scores (more negative = more anomalous)
        return -self.model.decision_function(data)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        predictions = self.model.predict(data)
        return (predictions == -1).astype(int)


class OneClassSVMDetector(AnomalyDetector):
    """
    One-Class SVM based anomaly detection.
    
    Learns a boundary around normal data in feature space.
    Points outside the boundary are anomalies.
    
    Advantages:
    - Can capture complex decision boundaries
    - Works well with kernel trick
    - Theoretically grounded
    
    Limitations:
    - Computationally expensive (O(n²) to O(n³))
    - Sensitive to kernel and parameters
    - Doesn't scale to very large datasets
    
    Best for:
    - Small to medium datasets
    - When decision boundary is complex
    - High-dimensional data with structure
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.05,
        gamma: str = 'scale'
    ):
        """
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            nu: Upper bound on fraction of outliers
            gamma: Kernel coefficient
        """
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'OneClassSVMDetector':
        """Fit One-Class SVM to training data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale data for better SVM performance
        data_scaled = self.scaler.fit_transform(data)
        self.model.fit(data_scaled)
        self.is_fitted = True
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly scores."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data_scaled = self.scaler.transform(data)
        return -self.model.decision_function(data_scaled)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data_scaled = self.scaler.transform(data)
        predictions = self.model.predict(data_scaled)
        return (predictions == -1).astype(int)


class DBSCANDetector(AnomalyDetector):
    """
    DBSCAN-based anomaly detection.
    
    Points that don't belong to any cluster (noise points) are anomalies.
    
    Advantages:
    - No assumption on number of clusters
    - Finds arbitrarily shaped clusters
    - Noise points are natural anomalies
    
    Limitations:
    - Sensitive to eps and min_samples
    - Struggles with varying densities
    - May mark too many or too few as noise
    
    Best for:
    - Clustered data
    - When anomalies are isolated points
    - Discovering natural groupings
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5
    ):
        """
        Args:
            eps: Maximum distance for neighborhood
            min_samples: Minimum points to form cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.labels_ = None
    
    def fit(self, data: np.ndarray) -> 'DBSCANDetector':
        """Fit DBSCAN to data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.fit_transform(data)
        
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = model.fit_predict(data_scaled)
        
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Score based on cluster membership.
        
        For new data, we'd need a different approach.
        Here we return binary scores from fit.
        """
        # For simplicity, return 1 for noise (-1 label), 0 for clustered
        if self.labels_ is not None:
            return (self.labels_ == -1).astype(float)
        return np.zeros(len(data))
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict based on cluster membership."""
        return (self.labels_ == -1).astype(int) if self.labels_ is not None else np.zeros(len(data), dtype=int)


class SimpleAutoencoderDetector(AnomalyDetector):
    """
    Simple Autoencoder-based anomaly detection.
    
    Learns to compress and reconstruct normal data.
    High reconstruction error indicates anomaly.
    
    Note: This is a simplified numpy-only implementation.
    For production, use PyTorch or TensorFlow.
    
    Advantages:
    - Can learn complex patterns
    - Captures nonlinear relationships
    - Dimensionality reduction built-in
    
    Limitations:
    - Requires more data for training
    - Can overfit
    - Hyperparameter sensitive
    
    Best for:
    - High-dimensional data
    - Complex patterns
    - When sufficient training data available
    """
    
    def __init__(
        self,
        encoding_dim: int = 5,
        threshold_percentile: float = 95,
        learning_rate: float = 0.01,
        epochs: int = 100
    ):
        """
        Args:
            encoding_dim: Dimension of encoded representation
            threshold_percentile: Percentile for anomaly threshold
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        """
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Weights (initialized during fit)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        self.scaler = StandardScaler()
        self.threshold = None
    
    def _relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)
    
    def _forward(self, X):
        """Forward pass."""
        # Encode
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        
        # Decode
        z2 = a1 @ self.W2 + self.b2
        output = z2  # Linear output for reconstruction
        
        return output, a1, z1
    
    def fit(self, data: np.ndarray) -> 'SimpleAutoencoderDetector':
        """Train autoencoder on normal data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale data
        X = self.scaler.fit_transform(data)
        n_samples, n_features = X.shape
        
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(n_features, self.encoding_dim) * 0.1
        self.b1 = np.zeros(self.encoding_dim)
        self.W2 = np.random.randn(self.encoding_dim, n_features) * 0.1
        self.b2 = np.zeros(n_features)
        
        # Training loop (simple gradient descent)
        for epoch in range(self.epochs):
            # Forward pass
            output, a1, z1 = self._forward(X)
            
            # Compute loss (MSE)
            loss = np.mean((output - X) ** 2)
            
            # Backward pass
            d_output = 2 * (output - X) / n_samples
            
            # Gradients for decoder
            d_W2 = a1.T @ d_output
            d_b2 = np.sum(d_output, axis=0)
            
            # Gradients for encoder
            d_a1 = d_output @ self.W2.T
            d_z1 = d_a1 * self._relu_derivative(z1)
            d_W1 = X.T @ d_z1
            d_b1 = np.sum(d_z1, axis=0)
            
            # Update weights
            self.W1 -= self.learning_rate * d_W1
            self.b1 -= self.learning_rate * d_b1
            self.W2 -= self.learning_rate * d_W2
            self.b2 -= self.learning_rate * d_b2
        
        # Set anomaly threshold based on reconstruction errors
        reconstruction_errors = self._get_reconstruction_error(X)
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        return self
    
    def _get_reconstruction_error(self, X):
        """Calculate reconstruction error."""
        output, _, _ = self._forward(X)
        return np.mean((output - X) ** 2, axis=1)
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return reconstruction error as anomaly score."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        X = self.scaler.transform(data)
        return self._get_reconstruction_error(X)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error threshold."""
        scores = self.score(data)
        return (scores > self.threshold).astype(int)


# =============================================================================
# SECTION 3: HYBRID AND ENSEMBLE APPROACHES
# =============================================================================

class EnsembleDetector(AnomalyDetector):
    """
    Ensemble anomaly detector combining multiple methods.
    
    Strategies:
    - voting: Majority vote across detectors
    - average: Average scores across detectors
    - max: Maximum score across detectors
    
    Advantages:
    - More robust than single detector
    - Reduces false positives
    - Combines strengths of different methods
    
    Best for:
    - Production systems
    - When no single method is clearly best
    - Critical applications requiring reliability
    """
    
    def __init__(
        self,
        detectors: List[AnomalyDetector],
        strategy: str = 'voting',
        weights: List[float] = None
    ):
        """
        Args:
            detectors: List of anomaly detectors
            strategy: 'voting', 'average', or 'max'
            weights: Optional weights for each detector
        """
        self.detectors = detectors
        self.strategy = strategy
        self.weights = weights or [1.0] * len(detectors)
    
    def fit(self, data: np.ndarray) -> 'EnsembleDetector':
        """Fit all detectors."""
        for detector in self.detectors:
            detector.fit(data)
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Combined anomaly scores.
        
        For 'voting': returns fraction of detectors predicting anomaly
        For 'average': weighted average of normalized scores
        For 'max': maximum score across detectors
        """
        if self.strategy == 'voting':
            predictions = [d.predict(data) for d in self.detectors]
            return np.average(predictions, axis=0, weights=self.weights)
        
        elif self.strategy == 'average':
            scores = []
            for d in self.detectors:
                s = d.score(data)
                # Normalize scores to [0, 1]
                s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10)
                scores.append(s_norm)
            return np.average(scores, axis=0, weights=self.weights)
        
        elif self.strategy == 'max':
            scores = []
            for d in self.detectors:
                s = d.score(data)
                s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10)
                scores.append(s_norm)
            return np.max(scores, axis=0)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies using ensemble."""
        scores = self.score(data)
        
        if self.strategy == 'voting':
            # Majority vote (> 0.5)
            return (scores > 0.5).astype(int)
        else:
            # For score-based strategies, use threshold
            threshold = 0.5
            return (scores > threshold).astype(int)


class TieredDetector(AnomalyDetector):
    """
    Tiered detection system.
    
    Uses fast statistical methods as first tier,
    then applies ML methods only on flagged points.
    
    Advantages:
    - Computational efficiency
    - Best of both worlds
    - Reduces ML inference costs
    
    Best for:
    - High-volume streams
    - When ML inference is expensive
    - Production systems with latency constraints
    """
    
    def __init__(
        self,
        fast_detector: AnomalyDetector,
        precise_detector: AnomalyDetector,
        fast_threshold: float = 2.0
    ):
        """
        Args:
            fast_detector: Quick statistical detector (first tier)
            precise_detector: More accurate ML detector (second tier)
            fast_threshold: Score threshold to pass to second tier
        """
        self.fast_detector = fast_detector
        self.precise_detector = precise_detector
        self.fast_threshold = fast_threshold
    
    def fit(self, data: np.ndarray) -> 'TieredDetector':
        """Fit both detectors."""
        self.fast_detector.fit(data)
        self.precise_detector.fit(data)
        return self
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Tiered scoring.
        
        Returns fast detector score for most points,
        precise detector score for suspicious points.
        """
        fast_scores = self.fast_detector.score(data)
        
        # Find suspicious points
        suspicious_mask = fast_scores > self.fast_threshold
        
        # Apply precise detector only to suspicious points
        final_scores = fast_scores.copy()
        if np.any(suspicious_mask):
            suspicious_data = data[suspicious_mask] if data.ndim == 1 else data[suspicious_mask, :]
            precise_scores = self.precise_detector.score(suspicious_data)
            final_scores[suspicious_mask] = precise_scores
        
        return final_scores
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Tiered prediction.
        
        First tier screens, second tier confirms.
        """
        fast_predictions = self.fast_detector.predict(data)
        
        # Only run precise detector on fast positives
        final_predictions = np.zeros_like(fast_predictions)
        suspicious_indices = np.where(fast_predictions == 1)[0]
        
        if len(suspicious_indices) > 0:
            suspicious_data = data[suspicious_indices] if data.ndim == 1 else data[suspicious_indices, :]
            precise_predictions = self.precise_detector.predict(suspicious_data)
            final_predictions[suspicious_indices] = precise_predictions
        
        return final_predictions
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return {
            'tier1_detector': type(self.fast_detector).__name__,
            'tier2_detector': type(self.precise_detector).__name__,
            'fast_threshold': self.fast_threshold
        }


# =============================================================================
# SECTION 4: EVALUATION AND COMPARISON
# =============================================================================

@dataclass
class DetectorEvaluation:
    """Evaluation results for an anomaly detector."""
    detector_name: str
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    detection_latency_ms: float
    predictions: np.ndarray = None
    scores: np.ndarray = None


class DetectorEvaluator:
    """
    Evaluate and compare anomaly detectors.
    
    Metrics:
    - Precision: True positives / Predicted positives
    - Recall: True positives / Actual positives
    - F1 Score: Harmonic mean of precision and recall
    - False Positive Rate: False positives / Actual negatives
    """
    
    @staticmethod
    def evaluate(
        detector: AnomalyDetector,
        data: np.ndarray,
        labels: np.ndarray
    ) -> DetectorEvaluation:
        """
        Evaluate a detector on labeled data.
        
        Args:
            detector: Fitted anomaly detector
            data: Test data
            labels: True labels (1=anomaly, 0=normal)
            
        Returns:
            DetectorEvaluation with metrics
        """
        import time
        
        # Time the prediction
        start = time.time()
        predictions = detector.predict(data)
        scores = detector.score(data)
        latency = (time.time() - start) * 1000 / len(data)  # ms per point
        
        # Calculate metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
        
        # False positive rate
        true_negatives = np.sum((labels == 0) & (predictions == 0))
        false_positives = np.sum((labels == 0) & (predictions == 1))
        total_negatives = true_negatives + false_positives
        fpr = false_positives / max(total_negatives, 1)
        
        return DetectorEvaluation(
            detector_name=type(detector).__name__,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            detection_latency_ms=latency,
            predictions=predictions,
            scores=scores
        )
    
    @staticmethod
    def compare_detectors(
        detectors: Dict[str, AnomalyDetector],
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, DetectorEvaluation]:
        """
        Compare multiple detectors on the same data.
        
        Args:
            detectors: Dictionary of name -> detector
            data: Test data
            labels: True labels
            
        Returns:
            Dictionary of name -> evaluation results
        """
        results = {}
        for name, detector in detectors.items():
            results[name] = DetectorEvaluator.evaluate(detector, data, labels)
        return results
    
    @staticmethod
    def print_comparison(results: Dict[str, DetectorEvaluation]):
        """Print formatted comparison table."""
        print("\n" + "="*80)
        print("DETECTOR COMPARISON RESULTS")
        print("="*80)
        print(f"{'Detector':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10} {'Latency':>12}")
        print("-"*80)
        
        for name, eval_result in sorted(results.items(), key=lambda x: x[1].f1_score, reverse=True):
            print(f"{eval_result.detector_name:<30} "
                  f"{eval_result.precision:>10.3f} "
                  f"{eval_result.recall:>10.3f} "
                  f"{eval_result.f1_score:>10.3f} "
                  f"{eval_result.false_positive_rate:>10.3f} "
                  f"{eval_result.detection_latency_ms:>10.3f}ms")
        
        print("-"*80)


# =============================================================================
# SECTION 5: THRESHOLD OPTIMIZATION
# =============================================================================

class ThresholdOptimizer:
    """
    Optimize detection thresholds for different objectives.
    
    Strategies:
    - Maximize F1 score
    - Fixed false positive rate
    - Fixed recall
    - Percentile-based
    """
    
    @staticmethod
    def optimize_f1(
        scores: np.ndarray,
        labels: np.ndarray,
        n_thresholds: int = 100
    ) -> Tuple[float, float]:
        """
        Find threshold that maximizes F1 score.
        
        Returns:
            (optimal_threshold, best_f1_score)
        """
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
        best_f1 = 0.0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1 = f1_score(labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    @staticmethod
    def optimize_fpr(
        scores: np.ndarray,
        labels: np.ndarray,
        target_fpr: float = 0.01
    ) -> float:
        """
        Find threshold that achieves target false positive rate.
        
        Returns:
            threshold achieving closest to target FPR
        """
        # Sort scores from normal points
        normal_scores = scores[labels == 0]
        
        # Threshold at (1 - target_fpr) percentile of normal scores
        threshold = np.percentile(normal_scores, (1 - target_fpr) * 100)
        
        return threshold
    
    @staticmethod
    def optimize_recall(
        scores: np.ndarray,
        labels: np.ndarray,
        target_recall: float = 0.95
    ) -> float:
        """
        Find threshold that achieves target recall.
        
        Returns:
            threshold achieving at least target recall
        """
        # Sort scores from anomaly points
        anomaly_scores = scores[labels == 1]
        
        if len(anomaly_scores) == 0:
            return np.max(scores)
        
        # Threshold at (1 - target_recall) percentile of anomaly scores
        threshold = np.percentile(anomaly_scores, (1 - target_recall) * 100)
        
        return threshold


# =============================================================================
# SECTION 6: DEMONSTRATION
# =============================================================================

def demo_statistical_methods():
    """Demonstrate statistical anomaly detection methods."""
    print("\n" + "="*70)
    print("SECTION 1: STATISTICAL ANOMALY DETECTION METHODS")
    print("="*70)
    
    # Generate test data
    data, timestamps, labels = generate_test_data(n_points=500, anomaly_fraction=0.05)
    
    print(f"\nGenerated {len(data)} data points with {int(labels.sum())} anomalies ({labels.mean()*100:.1f}%)")
    
    # Split into train/test
    split_idx = int(len(data) * 0.7)
    train_data, test_data = data[:split_idx], data[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    # Test each statistical method
    statistical_detectors = {
        'Z-Score (3σ)': ZScoreDetector(threshold=3.0),
        'Z-Score (2σ)': ZScoreDetector(threshold=2.0),
        'Percentile (1-99)': PercentileDetector(lower_percentile=1, upper_percentile=99),
        'Percentile (5-95)': PercentileDetector(lower_percentile=5, upper_percentile=95),
        'IQR (1.5x)': IQRDetector(k=1.5),
        'IQR (3.0x)': IQRDetector(k=3.0),
        'Moving Avg (SMA)': MovingAverageDetector(window_size=20, method='sma'),
        'Moving Avg (EMA)': MovingAverageDetector(window_size=20, method='ema'),
        'Seasonal Decomp': SeasonalDecompositionDetector(period=100),
    }
    
    # Fit and evaluate
    print("\n--- Statistical Methods Evaluation ---")
    results = {}
    for name, detector in statistical_detectors.items():
        detector.fit(train_data)
        results[name] = DetectorEvaluator.evaluate(detector, test_data, test_labels)
    
    DetectorEvaluator.print_comparison(results)
    
    return statistical_detectors, data, labels


def demo_ml_methods():
    """Demonstrate machine learning anomaly detection methods."""
    print("\n" + "="*70)
    print("SECTION 2: MACHINE LEARNING ANOMALY DETECTION METHODS")
    print("="*70)
    
    # Generate test data
    data, timestamps, labels = generate_test_data(n_points=500, anomaly_fraction=0.05)
    
    # Split into train/test
    split_idx = int(len(data) * 0.7)
    train_data, test_data = data[:split_idx], data[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    # Test ML methods
    ml_detectors = {
        'Isolation Forest': IsolationForestDetector(contamination=0.05),
        'LOF (k=20)': LOFDetector(n_neighbors=20, contamination=0.05),
        'One-Class SVM': OneClassSVMDetector(nu=0.05),
        'DBSCAN': DBSCANDetector(eps=0.5, min_samples=5),
        'Autoencoder': SimpleAutoencoderDetector(encoding_dim=3, epochs=50),
    }
    
    # Fit and evaluate
    print("\n--- ML Methods Evaluation ---")
    results = {}
    for name, detector in ml_detectors.items():
        print(f"Training {name}...")
        detector.fit(train_data)
        results[name] = DetectorEvaluator.evaluate(detector, test_data, test_labels)
    
    DetectorEvaluator.print_comparison(results)
    
    return ml_detectors


def demo_hybrid_methods():
    """Demonstrate hybrid and ensemble approaches."""
    print("\n" + "="*70)
    print("SECTION 3: HYBRID AND ENSEMBLE APPROACHES")
    print("="*70)
    
    # Generate test data
    data, timestamps, labels = generate_test_data(n_points=500, anomaly_fraction=0.05)
    
    # Split into train/test
    split_idx = int(len(data) * 0.7)
    train_data, test_data = data[:split_idx], data[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    # Create ensemble with statistical and ML methods
    print("\n--- Ensemble Detection ---")
    ensemble_voting = EnsembleDetector(
        detectors=[
            ZScoreDetector(threshold=2.5),
            IQRDetector(k=1.5),
            IsolationForestDetector(contamination=0.05),
        ],
        strategy='voting'
    )
    
    ensemble_average = EnsembleDetector(
        detectors=[
            ZScoreDetector(threshold=2.5),
            IQRDetector(k=1.5),
            IsolationForestDetector(contamination=0.05),
        ],
        strategy='average'
    )
    
    # Create tiered detector
    print("--- Tiered Detection ---")
    tiered = TieredDetector(
        fast_detector=ZScoreDetector(threshold=2.0),
        precise_detector=IsolationForestDetector(contamination=0.05),
        fast_threshold=1.5
    )
    
    hybrid_detectors = {
        'Ensemble (Voting)': ensemble_voting,
        'Ensemble (Average)': ensemble_average,
        'Tiered (Z-Score → IF)': tiered,
    }
    
    # Evaluate
    results = {}
    for name, detector in hybrid_detectors.items():
        print(f"Training {name}...")
        detector.fit(train_data)
        results[name] = DetectorEvaluator.evaluate(detector, test_data, test_labels)
    
    DetectorEvaluator.print_comparison(results)
    
    return hybrid_detectors


def demo_threshold_optimization():
    """Demonstrate threshold optimization strategies."""
    print("\n" + "="*70)
    print("SECTION 4: THRESHOLD OPTIMIZATION")
    print("="*70)
    
    # Generate test data
    data, timestamps, labels = generate_test_data(n_points=500, anomaly_fraction=0.05)
    
    # Train a detector
    detector = IsolationForestDetector(contamination=0.1)
    detector.fit(data)
    scores = detector.score(data)
    
    print("\n--- Threshold Optimization Strategies ---")
    
    # Strategy 1: Maximize F1
    opt_threshold_f1, best_f1 = ThresholdOptimizer.optimize_f1(scores, labels)
    print(f"\n1. Maximize F1 Score:")
    print(f"   Optimal threshold: {opt_threshold_f1:.3f}")
    print(f"   Best F1 score: {best_f1:.3f}")
    
    # Strategy 2: Target FPR
    opt_threshold_fpr = ThresholdOptimizer.optimize_fpr(scores, labels, target_fpr=0.01)
    predictions_fpr = (scores > opt_threshold_fpr).astype(int)
    actual_fpr = np.sum((labels == 0) & (predictions_fpr == 1)) / max(np.sum(labels == 0), 1)
    print(f"\n2. Target FPR = 1%:")
    print(f"   Threshold: {opt_threshold_fpr:.3f}")
    print(f"   Actual FPR: {actual_fpr*100:.2f}%")
    
    # Strategy 3: Target Recall
    opt_threshold_recall = ThresholdOptimizer.optimize_recall(scores, labels, target_recall=0.9)
    predictions_recall = (scores > opt_threshold_recall).astype(int)
    actual_recall = recall_score(labels, predictions_recall, zero_division=0)
    print(f"\n3. Target Recall = 90%:")
    print(f"   Threshold: {opt_threshold_recall:.3f}")
    print(f"   Actual Recall: {actual_recall*100:.2f}%")
    
    # Strategy 4: Percentile-based
    percentile_threshold = np.percentile(scores, 95)
    predictions_pct = (scores > percentile_threshold).astype(int)
    print(f"\n4. 95th Percentile:")
    print(f"   Threshold: {percentile_threshold:.3f}")
    print(f"   Anomalies flagged: {predictions_pct.sum()} ({predictions_pct.mean()*100:.1f}%)")


def demo_decision_framework():
    """Demonstrate the decision framework for choosing detectors."""
    print("\n" + "="*70)
    print("SECTION 5: DECISION FRAMEWORK")
    print("="*70)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                    ANOMALY DETECTION DECISION TREE                    ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  START: What does your data look like?                               ║
    ║         │                                                            ║
    ║         ├── Simple, stable, ~normal distribution?                    ║
    ║         │   └── ✅ Use Z-Score or Percentile methods                 ║
    ║         │       • Fast, interpretable, no training needed            ║
    ║         │                                                            ║
    ║         ├── Has seasonal patterns?                                   ║
    ║         │   └── ✅ Use Seasonal Decomposition                        ║
    ║         │       • Or Moving Average with appropriate window          ║
    ║         │                                                            ║
    ║         ├── High-dimensional / multivariate?                         ║
    ║         │   └── ✅ Use Isolation Forest or Autoencoder              ║
    ║         │       • IF: Fast, scalable                                 ║
    ║         │       • AE: Captures complex patterns                      ║
    ║         │                                                            ║
    ║         ├── Data has varying densities / clusters?                   ║
    ║         │   └── ✅ Use LOF or DBSCAN                                 ║
    ║         │       • LOF: Density-aware local anomalies                 ║
    ║         │       • DBSCAN: Cluster-based detection                    ║
    ║         │                                                            ║
    ║         ├── Need high reliability / production system?               ║
    ║         │   └── ✅ Use Ensemble or Tiered approach                   ║
    ║         │       • Combines strengths of multiple methods             ║
    ║         │       • Reduces false positives                            ║
    ║         │                                                            ║
    ║         └── Limited data / cold start?                               ║
    ║             └── ✅ Start with Statistical methods                    ║
    ║                 • Z-Score, IQR work from day one                     ║
    ║                 • Add ML methods as data accumulates                 ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n--- Quick Reference: Method Comparison ---")
    print("""
    ┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    │ Method              │ Speed       │ Accuracy    │ Interpret.  │ Data Needed │
    ├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
    │ Z-Score             │ ⚡⚡⚡⚡⚡     │ ⭐⭐          │ ⭐⭐⭐⭐⭐     │ Minimal     │
    │ Percentile          │ ⚡⚡⚡⚡⚡     │ ⭐⭐⭐        │ ⭐⭐⭐⭐⭐     │ Minimal     │
    │ IQR                 │ ⚡⚡⚡⚡⚡     │ ⭐⭐          │ ⭐⭐⭐⭐⭐     │ Minimal     │
    │ Moving Average      │ ⚡⚡⚡⚡      │ ⭐⭐⭐        │ ⭐⭐⭐⭐      │ Some        │
    │ Seasonal Decomp     │ ⚡⚡⚡       │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐      │ Moderate    │
    │ Isolation Forest    │ ⚡⚡⚡⚡      │ ⭐⭐⭐⭐      │ ⭐⭐⭐        │ Moderate    │
    │ LOF                 │ ⚡⚡         │ ⭐⭐⭐⭐      │ ⭐⭐⭐        │ Moderate    │
    │ One-Class SVM       │ ⚡          │ ⭐⭐⭐⭐      │ ⭐⭐          │ Moderate    │
    │ Autoencoder         │ ⚡⚡         │ ⭐⭐⭐⭐⭐    │ ⭐⭐          │ Large       │
    │ Ensemble            │ ⚡⚡⚡       │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐        │ Varies      │
    └─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all demonstrations."""
    print("="*70)
    print("CHAPTER 7: STATISTICAL VS. ML-BASED APPROACHES")
    print("Anomaly Detection Method Comparison")
    print("="*70)
    
    # Demonstrate each category
    demo_statistical_methods()
    demo_ml_methods()
    demo_hybrid_methods()
    demo_threshold_optimization()
    demo_decision_framework()
    
    print("\n" + "="*70)
    print("✅ All demonstrations completed!")
    print("="*70)
    
    print("""
    KEY TAKEAWAYS:
    
    1. Statistical methods (Z-score, IQR, Percentile) are fast, interpretable,
       and work without training data - use them for simple cases and baselines.
    
    2. ML methods (Isolation Forest, LOF, Autoencoder) capture complex patterns
       but need training data and careful tuning.
    
    3. Hybrid approaches combine strengths: use statistical methods for
       screening, ML methods for confirmation.
    
    4. Threshold optimization is critical - different strategies suit
       different business requirements (precision vs recall).
    
    5. Start simple, add complexity as needed. Z-score might be enough!
    """)


if __name__ == "__main__":
    main()
