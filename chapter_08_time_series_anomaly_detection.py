"""
Chapter 8: Time Series Anomaly Detection
==========================================

Production-ready code demonstrating time series anomaly detection techniques:
- Statistical methods (moving averages, STL decomposition, exponential smoothing)
- Forecasting-based detection (seasonal models, trend analysis)
- Machine learning approaches (Isolation Forest for time series)
- Deep learning methods (LSTM-based anomaly detection)
- AWS Bedrock integration for analysis and root cause explanation
- Multi-variate time series anomaly detection
- Real-time streaming detection

Prerequisites:
    pip install boto3 numpy pandas scikit-learn statsmodels tensorflow

AWS Configuration:
    Ensure you have AWS credentials configured with access to Amazon Bedrock.
    Enable the following models in your Bedrock console:
    - anthropic.claude-3-sonnet-20240229-v1:0 (for analysis/explanation)

Note: This code demonstrates multiple approaches. In production, choose the methods
      that best fit your data characteristics and requirements.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import boto3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Some statistical methods will be disabled.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM detector will be disabled.")


# =============================================================================
# CONFIGURATION
# =============================================================================

LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

def get_bedrock_client(region_name: str = "us-east-1"):
    """Create a Bedrock runtime client."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimePoint:
    """A single time series data point."""
    timestamp: datetime
    value: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Anomaly:
    """Detected anomaly information."""
    timestamp: datetime
    value: float
    expected_value: float
    anomaly_score: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    detection_method: str
    confidence: float
    explanation: Optional[str] = None


@dataclass
class TimeSeriesMetadata:
    """Metadata about a time series."""
    name: str
    frequency: str  # 'minute', 'hour', 'day', etc.
    has_trend: bool
    has_seasonality: bool
    seasonal_period: Optional[int] = None  # e.g., 24 for hourly data with daily pattern
    unit: Optional[str] = None


# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_sample_time_series(
    start_date: datetime,
    periods: int = 1000,
    frequency: str = 'H',
    trend: bool = True,
    seasonality: bool = True,
    noise_level: float = 0.1,
    anomaly_rate: float = 0.02
) -> pd.DataFrame:
    """
    Generate realistic sample time series data with optional trend, seasonality, and anomalies.
    
    Args:
        start_date: Starting timestamp
        periods: Number of data points
        frequency: Pandas frequency string ('H' for hourly, 'D' for daily, etc.)
        trend: Whether to include upward trend
        seasonality: Whether to include seasonal pattern
        noise_level: Amount of random noise (as fraction of signal)
        anomaly_rate: Fraction of points that should be anomalous
        
    Returns:
        DataFrame with timestamp and value columns
    """
    # Generate timestamps
    timestamps = pd.date_range(start=start_date, periods=periods, freq=frequency)
    
    # Base value
    base = 100.0
    values = np.ones(periods) * base
    
    # Add trend if requested
    if trend:
        trend_component = np.linspace(0, 20, periods)
        values += trend_component
    
    # Add seasonality if requested
    if seasonality:
        # Daily pattern (assuming hourly frequency)
        if frequency == 'H':
            seasonal_component = 15 * np.sin(2 * np.pi * np.arange(periods) / 24)
        # Weekly pattern (assuming daily frequency)
        elif frequency == 'D':
            seasonal_component = 10 * np.sin(2 * np.pi * np.arange(periods) / 7)
        else:
            seasonal_component = 10 * np.sin(2 * np.pi * np.arange(periods) / 12)
        values += seasonal_component
    
    # Add noise
    noise = np.random.normal(0, noise_level * base, periods)
    values += noise
    
    # Inject anomalies
    n_anomalies = int(periods * anomaly_rate)
    anomaly_indices = np.random.choice(periods, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Random spike or drop
        if np.random.random() > 0.5:
            values[idx] += np.random.uniform(30, 50)  # Spike
        else:
            values[idx] -= np.random.uniform(20, 40)  # Drop
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    return df


# =============================================================================
# STATISTICAL METHODS
# =============================================================================

class MovingAverageDetector:
    """
    Simple anomaly detection using moving average and standard deviation.
    Works well for stable metrics without trend or seasonality.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        threshold_std: float = 3.0
    ):
        """
        Initialize moving average detector.
        
        Args:
            window_size: Number of points to include in moving average
            threshold_std: Number of standard deviations for anomaly threshold
        """
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.values_buffer = deque(maxlen=window_size)
    
    def detect(self, value: float) -> Tuple[bool, float, float]:
        """
        Detect if a value is anomalous.
        
        Args:
            value: New value to check
            
        Returns:
            Tuple of (is_anomaly, expected_value, anomaly_score)
        """
        # Need enough history
        if len(self.values_buffer) < self.window_size:
            self.values_buffer.append(value)
            return False, value, 0.0
        
        # Calculate moving average and std
        mean = np.mean(self.values_buffer)
        std = np.std(self.values_buffer)
        
        # Avoid division by zero
        if std < 1e-6:
            std = 1e-6
        
        # Calculate z-score
        z_score = abs(value - mean) / std
        
        # Check if anomalous
        is_anomaly = z_score > self.threshold_std
        
        # Add to buffer for next iteration
        self.values_buffer.append(value)
        
        return is_anomaly, mean, z_score
    
    def detect_batch(self, df: pd.DataFrame) -> List[Anomaly]:
        """
        Detect anomalies in a batch of data.
        
        Args:
            df: DataFrame with 'timestamp' and 'value' columns
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for _, row in df.iterrows():
            is_anomaly, expected, score = self.detect(row['value'])
            
            if is_anomaly:
                anomaly = Anomaly(
                    timestamp=row['timestamp'],
                    value=row['value'],
                    expected_value=expected,
                    anomaly_score=score,
                    severity=self._calculate_severity(score),
                    detection_method='moving_average',
                    confidence=min(score / 5.0, 1.0)  # Normalize to 0-1
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    @staticmethod
    def _calculate_severity(z_score: float) -> str:
        """Calculate severity based on z-score."""
        if z_score > 5:
            return 'critical'
        elif z_score > 4:
            return 'high'
        elif z_score > 3:
            return 'medium'
        else:
            return 'low'


class ExponentialSmoothingDetector:
    """
    Anomaly detection using exponential smoothing.
    Better than simple moving average for data with trends.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        threshold_factor: float = 3.0
    ):
        """
        Initialize exponential smoothing detector.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = more weight to recent values
            threshold_factor: Multiplier for deviation threshold
        """
        self.alpha = alpha
        self.threshold_factor = threshold_factor
        self.smoothed_value = None
        self.smoothed_deviation = None
    
    def detect(self, value: float) -> Tuple[bool, float, float]:
        """
        Detect if a value is anomalous using exponential smoothing.
        
        Args:
            value: New value to check
            
        Returns:
            Tuple of (is_anomaly, expected_value, anomaly_score)
        """
        # Initialize on first value
        if self.smoothed_value is None:
            self.smoothed_value = value
            self.smoothed_deviation = 0.0
            return False, value, 0.0
        
        # Calculate prediction error
        error = abs(value - self.smoothed_value)
        
        # Update smoothed deviation
        if self.smoothed_deviation == 0:
            self.smoothed_deviation = error
        else:
            self.smoothed_deviation = (
                self.alpha * error + 
                (1 - self.alpha) * self.smoothed_deviation
            )
        
        # Check if anomalous
        threshold = self.threshold_factor * self.smoothed_deviation
        is_anomaly = error > threshold
        
        # Calculate anomaly score
        if self.smoothed_deviation > 0:
            anomaly_score = error / self.smoothed_deviation
        else:
            anomaly_score = 0.0
        
        # Update smoothed value
        self.smoothed_value = (
            self.alpha * value + 
            (1 - self.alpha) * self.smoothed_value
        )
        
        return is_anomaly, self.smoothed_value, anomaly_score


class STLDecompositionDetector:
    """
    Seasonal-Trend decomposition using Loess (STL).
    Best for data with clear seasonal patterns and trends.
    Requires statsmodels.
    """
    
    def __init__(
        self,
        seasonal_period: int = 24,
        threshold_std: float = 3.0
    ):
        """
        Initialize STL decomposition detector.
        
        Args:
            seasonal_period: Length of seasonal cycle (e.g., 24 for hourly data with daily pattern)
            threshold_std: Number of standard deviations for threshold
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for STL decomposition")
        
        self.seasonal_period = seasonal_period
        self.threshold_std = threshold_std
    
    def detect_batch(self, df: pd.DataFrame) -> List[Anomaly]:
        """
        Detect anomalies using STL decomposition.
        
        Args:
            df: DataFrame with 'timestamp' and 'value' columns
            
        Returns:
            List of detected anomalies
        """
        # Need enough data for decomposition
        min_points = self.seasonal_period * 2
        if len(df) < min_points:
            print(f"Warning: Need at least {min_points} points for STL decomposition")
            return []
        
        # Perform STL decomposition
        try:
            stl = STL(df['value'], seasonal=self.seasonal_period)
            result = stl.fit()
            
            # Extract components
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            # Detect anomalies in residuals
            residual_mean = np.mean(residual)
            residual_std = np.std(residual)
            
            threshold = self.threshold_std * residual_std
            
            anomalies = []
            for idx, row in df.iterrows():
                if abs(residual[idx] - residual_mean) > threshold:
                    expected = trend[idx] + seasonal[idx]
                    z_score = abs(residual[idx] - residual_mean) / residual_std
                    
                    anomaly = Anomaly(
                        timestamp=row['timestamp'],
                        value=row['value'],
                        expected_value=expected,
                        anomaly_score=z_score,
                        severity=self._calculate_severity(z_score),
                        detection_method='stl_decomposition',
                        confidence=min(z_score / 5.0, 1.0)
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            print(f"Error in STL decomposition: {e}")
            return []
    
    @staticmethod
    def _calculate_severity(z_score: float) -> str:
        """Calculate severity based on z-score."""
        if z_score > 5:
            return 'critical'
        elif z_score > 4:
            return 'high'
        elif z_score > 3:
            return 'medium'
        else:
            return 'low'


# =============================================================================
# MACHINE LEARNING METHODS
# =============================================================================

class IsolationForestTimeSeriesDetector:
    """
    Isolation Forest adapted for time series data.
    Works by creating features from temporal context.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        contamination: float = 0.05,
        n_estimators: int = 100
    ):
        """
        Initialize Isolation Forest time series detector.
        
        Args:
            window_size: Number of previous points to use as features
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
        """
        self.window_size = window_size
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create features for each time point using sliding window.
        
        Features include:
        - Recent values (window_size points)
        - Rolling mean and std
        - Hour of day (if timestamp available)
        - Day of week (if timestamp available)
        """
        features_list = []
        
        values = df['value'].values
        
        for i in range(self.window_size, len(values)):
            # Recent values
            window = values[i-self.window_size:i]
            
            # Statistical features
            window_mean = np.mean(window)
            window_std = np.std(window)
            window_min = np.min(window)
            window_max = np.max(window)
            
            # Current value relative to window
            current_value = values[i]
            relative_position = (current_value - window_mean) / (window_std + 1e-6)
            
            # Time-based features if timestamp available
            if 'timestamp' in df.columns:
                ts = df.iloc[i]['timestamp']
                hour = ts.hour
                day_of_week = ts.dayofweek
                time_features = [hour, day_of_week]
            else:
                time_features = []
            
            # Combine all features
            feature_vector = [
                current_value,
                window_mean,
                window_std,
                window_min,
                window_max,
                relative_position
            ] + time_features
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def fit(self, df: pd.DataFrame):
        """
        Train the Isolation Forest model on normal data.
        
        Args:
            df: DataFrame with 'timestamp' and 'value' columns
        """
        features = self._create_features(df)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        self.is_fitted = True
    
    def detect_batch(self, df: pd.DataFrame) -> List[Anomaly]:
        """
        Detect anomalies in a batch of data.
        
        Args:
            df: DataFrame with 'timestamp' and 'value' columns
            
        Returns:
            List of detected anomalies
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detection")
        
        features = self._create_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Get anomaly predictions and scores
        predictions = self.model.predict(features_scaled)
        scores = self.model.score_samples(features_scaled)
        
        # Normalize scores to 0-1 range
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        anomalies = []
        
        # Start from window_size since we lose initial points in feature creation
        for i, pred in enumerate(predictions):
            if pred == -1:  # Anomaly
                idx = i + self.window_size
                row = df.iloc[idx]
                
                # Calculate expected value (mean of recent window)
                window_values = df.iloc[idx-self.window_size:idx]['value'].values
                expected = np.mean(window_values)
                
                anomaly = Anomaly(
                    timestamp=row['timestamp'],
                    value=row['value'],
                    expected_value=expected,
                    anomaly_score=1.0 - scores_normalized[i],
                    severity=self._calculate_severity(1.0 - scores_normalized[i]),
                    detection_method='isolation_forest',
                    confidence=1.0 - scores_normalized[i]
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    @staticmethod
    def _calculate_severity(score: float) -> str:
        """Calculate severity based on anomaly score."""
        if score > 0.9:
            return 'critical'
        elif score > 0.75:
            return 'high'
        elif score > 0.6:
            return 'medium'
        else:
            return 'low'


# =============================================================================
# DEEP LEARNING METHODS
# =============================================================================

class LSTMAutoencoderDetector:
    """
    LSTM-based autoencoder for anomaly detection.
    Learns to reconstruct normal patterns; poor reconstruction indicates anomaly.
    Requires TensorFlow.
    """
    
    def __init__(
        self,
        sequence_length: int = 24,
        encoding_dim: int = 16,
        threshold_percentile: float = 95.0
    ):
        """
        Initialize LSTM autoencoder detector.
        
        Args:
            sequence_length: Length of input sequences
            encoding_dim: Dimension of the encoded representation
            threshold_percentile: Percentile of reconstruction error for threshold
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM autoencoder")
        
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
    
    def _build_model(self):
        """Build the LSTM autoencoder model."""
        # Encoder
        encoder_inputs = keras.Input(shape=(self.sequence_length, 1))
        encoded = layers.LSTM(32, return_sequences=True)(encoder_inputs)
        encoded = layers.LSTM(self.encoding_dim)(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        decoded = layers.LSTM(self.encoding_dim, return_sequences=True)(decoded)
        decoded = layers.LSTM(32, return_sequences=True)(decoded)
        decoded = layers.TimeDistributed(layers.Dense(1))(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(encoder_inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sliding window sequences."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i+self.sequence_length])
        return np.array(sequences)
    
    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 0
    ):
        """
        Train the LSTM autoencoder on normal data.
        
        Args:
            df: DataFrame with 'value' column
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        # Prepare data
        values = df['value'].values.reshape(-1, 1)
        values_scaled = self.scaler.fit_transform(values)
        
        # Create sequences
        sequences = self._create_sequences(values_scaled)
        
        # Build and train model
        self.model = self._build_model()
        
        history = self.model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        # Calculate reconstruction errors on training data to set threshold
        reconstructions = self.model.predict(sequences, verbose=0)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        self.threshold = np.percentile(mse, self.threshold_percentile)
        
        return history
    
    def detect_batch(self, df: pd.DataFrame) -> List[Anomaly]:
        """
        Detect anomalies using reconstruction error.
        
        Args:
            df: DataFrame with 'timestamp' and 'value' columns
            
        Returns:
            List of detected anomalies
        """
        if self.model is None:
            raise ValueError("Model must be fitted before detection")
        
        # Prepare data
        values = df['value'].values.reshape(-1, 1)
        values_scaled = self.scaler.transform(values)
        
        # Create sequences
        sequences = self._create_sequences(values_scaled)
        
        # Get reconstructions
        reconstructions = self.model.predict(sequences, verbose=0)
        
        # Calculate reconstruction errors
        mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        
        # Detect anomalies
        anomalies = []
        
        for i, error in enumerate(mse):
            if error > self.threshold:
                # Index in original data
                idx = i + self.sequence_length - 1
                row = df.iloc[idx]
                
                # Calculate expected value (reconstruction)
                expected_scaled = reconstructions[i, -1, 0]
                expected = self.scaler.inverse_transform([[expected_scaled]])[0][0]
                
                # Normalize score
                score = min(error / (self.threshold + 1e-6), 10.0) / 10.0
                
                anomaly = Anomaly(
                    timestamp=row['timestamp'],
                    value=row['value'],
                    expected_value=expected,
                    anomaly_score=error,
                    severity=self._calculate_severity(score),
                    detection_method='lstm_autoencoder',
                    confidence=score
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    @staticmethod
    def _calculate_severity(score: float) -> str:
        """Calculate severity based on normalized score."""
        if score > 0.9:
            return 'critical'
        elif score > 0.7:
            return 'high'
        elif score > 0.5:
            return 'medium'
        else:
            return 'low'


# =============================================================================
# AWS BEDROCK INTEGRATION FOR ANALYSIS
# =============================================================================

def explain_anomalies_with_bedrock(
    anomalies: List[Anomaly],
    time_series_metadata: TimeSeriesMetadata,
    context: Optional[str] = None
) -> List[Anomaly]:
    """
    Use AWS Bedrock to generate explanations for detected anomalies.
    
    Args:
        anomalies: List of detected anomalies
        time_series_metadata: Metadata about the time series
        context: Optional additional context about the system
        
    Returns:
        List of anomalies with added explanations
    """
    if not anomalies:
        return anomalies
    
    try:
        bedrock_client = get_bedrock_client()
        
        for anomaly in anomalies:
            # Build prompt
            prompt = f"""Analyze this time series anomaly:

Metric: {time_series_metadata.name}
Timestamp: {anomaly.timestamp}
Observed Value: {anomaly.value:.2f} {time_series_metadata.unit or ''}
Expected Value: {anomaly.expected_value:.2f} {time_series_metadata.unit or ''}
Deviation: {abs(anomaly.value - anomaly.expected_value):.2f} ({((anomaly.value - anomaly.expected_value) / anomaly.expected_value * 100):.1f}%)
Severity: {anomaly.severity}
Detection Method: {anomaly.detection_method}

Time Series Characteristics:
- Has Trend: {time_series_metadata.has_trend}
- Has Seasonality: {time_series_metadata.has_seasonality}
- Seasonal Period: {time_series_metadata.seasonal_period or 'N/A'}

{f'Additional Context: {context}' if context else ''}

Provide a concise 2-3 sentence explanation of:
1. What this anomaly represents (spike, drop, shift, etc.)
2. Possible root causes given the context
3. Recommended next steps for investigation

Be specific and actionable."""

            # Call Bedrock
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = bedrock_client.invoke_model(
                modelId=LLM_MODEL_ID,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            explanation = response_body['content'][0]['text'].strip()
            
            # Add explanation to anomaly
            anomaly.explanation = explanation
    
    except Exception as e:
        print(f"Error calling Bedrock for anomaly explanation: {e}")
    
    return anomalies


# =============================================================================
# ENSEMBLE DETECTOR
# =============================================================================

class EnsembleTimeSeriesDetector:
    """
    Combines multiple detection methods for robust anomaly detection.
    Uses voting or weighted scoring to reduce false positives.
    """
    
    def __init__(
        self,
        detectors: List[Any],
        voting_threshold: int = 2,
        use_weighted: bool = False
    ):
        """
        Initialize ensemble detector.
        
        Args:
            detectors: List of detector instances
            voting_threshold: Minimum number of detectors that must agree
            use_weighted: Whether to use weighted scoring (by confidence)
        """
        self.detectors = detectors
        self.voting_threshold = voting_threshold
        self.use_weighted = use_weighted
    
    def detect_batch(self, df: pd.DataFrame) -> List[Anomaly]:
        """
        Detect anomalies using ensemble of methods.
        
        Args:
            df: DataFrame with 'timestamp' and 'value' columns
            
        Returns:
            List of detected anomalies with consensus
        """
        # Collect anomalies from each detector
        all_anomalies = {}  # timestamp -> list of anomalies
        
        for detector in self.detectors:
            try:
                detector_anomalies = detector.detect_batch(df)
                
                for anomaly in detector_anomalies:
                    ts = anomaly.timestamp
                    if ts not in all_anomalies:
                        all_anomalies[ts] = []
                    all_anomalies[ts].append(anomaly)
            
            except Exception as e:
                print(f"Error in detector {type(detector).__name__}: {e}")
                continue
        
        # Aggregate anomalies
        consensus_anomalies = []
        
        for timestamp, anomaly_list in all_anomalies.items():
            # Check voting threshold
            if len(anomaly_list) >= self.voting_threshold:
                # Combine information from multiple detectors
                if self.use_weighted:
                    # Weighted average by confidence
                    weights = [a.confidence for a in anomaly_list]
                    total_weight = sum(weights)
                    
                    avg_value = sum(a.value * a.confidence for a in anomaly_list) / total_weight
                    avg_expected = sum(a.expected_value * a.confidence for a in anomaly_list) / total_weight
                    avg_score = sum(a.anomaly_score * a.confidence for a in anomaly_list) / total_weight
                else:
                    # Simple average
                    avg_value = np.mean([a.value for a in anomaly_list])
                    avg_expected = np.mean([a.expected_value for a in anomaly_list])
                    avg_score = np.mean([a.anomaly_score for a in anomaly_list])
                
                # Determine severity (take worst)
                severities = [a.severity for a in anomaly_list]
                severity_order = ['low', 'medium', 'high', 'critical']
                max_severity = max(severities, key=lambda s: severity_order.index(s))
                
                # Combine detection methods
                methods = [a.detection_method for a in anomaly_list]
                
                consensus_anomaly = Anomaly(
                    timestamp=timestamp,
                    value=avg_value,
                    expected_value=avg_expected,
                    anomaly_score=avg_score,
                    severity=max_severity,
                    detection_method=f"ensemble({len(anomaly_list)}/{len(self.detectors)}): {', '.join(set(methods))}",
                    confidence=len(anomaly_list) / len(self.detectors)
                )
                
                consensus_anomalies.append(consensus_anomaly)
        
        return consensus_anomalies


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_anomaly_report(anomalies: List[Anomaly]):
    """Print a formatted report of detected anomalies."""
    if not anomalies:
        print("No anomalies detected.")
        return
    
    print(f"\n{'='*80}")
    print(f"ANOMALY DETECTION REPORT")
    print(f"{'='*80}")
    print(f"Total Anomalies Detected: {len(anomalies)}\n")
    
    # Sort by severity
    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    sorted_anomalies = sorted(anomalies, key=lambda a: severity_order[a.severity])
    
    for i, anomaly in enumerate(sorted_anomalies, 1):
        deviation = anomaly.value - anomaly.expected_value
        deviation_pct = (deviation / anomaly.expected_value * 100) if anomaly.expected_value != 0 else 0
        
        print(f"Anomaly #{i}")
        print(f"  Timestamp: {anomaly.timestamp}")
        print(f"  Severity: {anomaly.severity.upper()}")
        print(f"  Value: {anomaly.value:.2f}")
        print(f"  Expected: {anomaly.expected_value:.2f}")
        print(f"  Deviation: {deviation:+.2f} ({deviation_pct:+.1f}%)")
        print(f"  Score: {anomaly.anomaly_score:.2f}")
        print(f"  Method: {anomaly.detection_method}")
        print(f"  Confidence: {anomaly.confidence:.2%}")
        
        if anomaly.explanation:
            print(f"  Explanation: {anomaly.explanation}")
        
        print()


def compare_detectors(
    df: pd.DataFrame,
    detectors: Dict[str, Any]
) -> Dict[str, List[Anomaly]]:
    """
    Compare multiple detectors on the same data.
    
    Args:
        df: DataFrame with time series data
        detectors: Dictionary mapping detector name to detector instance
        
    Returns:
        Dictionary mapping detector name to detected anomalies
    """
    results = {}
    
    print("Comparing detectors...")
    print(f"Dataset: {len(df)} points\n")
    
    for name, detector in detectors.items():
        print(f"Running {name}...")
        try:
            anomalies = detector.detect_batch(df)
            results[name] = anomalies
            print(f"  Detected: {len(anomalies)} anomalies")
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = []
    
    return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Demonstrate time series anomaly detection."""
    print("Time Series Anomaly Detection Examples")
    print("=" * 80)
    
    # Generate sample data
    print("\n1. Generating sample time series data...")
    start_date = datetime(2024, 1, 1)
    df = generate_sample_time_series(
        start_date=start_date,
        periods=500,
        frequency='H',
        trend=True,
        seasonality=True,
        anomaly_rate=0.03
    )
    print(f"   Generated {len(df)} hourly data points")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Example 1: Simple Moving Average Detection
    print("\n2. Simple Moving Average Detection")
    print("-" * 80)
    ma_detector = MovingAverageDetector(window_size=24, threshold_std=3.0)
    ma_anomalies = ma_detector.detect_batch(df)
    print(f"   Detected {len(ma_anomalies)} anomalies")
    
    # Example 2: Exponential Smoothing
    print("\n3. Exponential Smoothing Detection")
    print("-" * 80)
    es_detector = ExponentialSmoothingDetector(alpha=0.3, threshold_factor=3.0)
    es_anomalies = es_detector.detect_batch(df)
    print(f"   Detected {len(es_anomalies)} anomalies")
    
    # Example 3: STL Decomposition (if available)
    if STATSMODELS_AVAILABLE:
        print("\n4. STL Decomposition Detection")
        print("-" * 80)
        stl_detector = STLDecompositionDetector(seasonal_period=24, threshold_std=3.0)
        stl_anomalies = stl_detector.detect_batch(df)
        print(f"   Detected {len(stl_anomalies)} anomalies")
    
    # Example 4: Isolation Forest
    print("\n5. Isolation Forest Detection")
    print("-" * 80)
    if_detector = IsolationForestTimeSeriesDetector(
        window_size=10,
        contamination=0.05
    )
    # Train on first 80% of data
    train_size = int(len(df) * 0.8)
    if_detector.fit(df[:train_size])
    if_anomalies = if_detector.detect_batch(df)
    print(f"   Detected {len(if_anomalies)} anomalies")
    
    # Example 5: LSTM Autoencoder (if TensorFlow available)
    if TENSORFLOW_AVAILABLE:
        print("\n6. LSTM Autoencoder Detection")
        print("-" * 80)
        lstm_detector = LSTMAutoencoderDetector(
            sequence_length=24,
            encoding_dim=16
        )
        # Train on first 80% of data
        lstm_detector.fit(df[:train_size], epochs=30, verbose=0)
        lstm_anomalies = lstm_detector.detect_batch(df)
        print(f"   Detected {len(lstm_anomalies)} anomalies")
    
    # Example 6: Ensemble Detection
    print("\n7. Ensemble Detection")
    print("-" * 80)
    
    # Create ensemble of statistical methods
    ensemble_detectors = [
        MovingAverageDetector(window_size=24),
        ExponentialSmoothingDetector(alpha=0.3)
    ]
    
    if STATSMODELS_AVAILABLE:
        ensemble_detectors.append(
            STLDecompositionDetector(seasonal_period=24)
        )
    
    ensemble = EnsembleTimeSeriesDetector(
        detectors=ensemble_detectors,
        voting_threshold=2
    )
    ensemble_anomalies = ensemble.detect_batch(df)
    print(f"   Detected {len(ensemble_anomalies)} anomalies with consensus")
    
    # Example 7: AWS Bedrock Explanation
    print("\n8. Adding AI-Powered Explanations (AWS Bedrock)")
    print("-" * 80)
    
    # Take top 3 anomalies for explanation
    top_anomalies = sorted(
        ensemble_anomalies,
        key=lambda a: a.anomaly_score,
        reverse=True
    )[:3]
    
    metadata = TimeSeriesMetadata(
        name="API Response Time",
        frequency="hour",
        has_trend=True,
        has_seasonality=True,
        seasonal_period=24,
        unit="ms"
    )
    
    print("   Requesting explanations from AWS Bedrock...")
    explained_anomalies = explain_anomalies_with_bedrock(
        top_anomalies,
        metadata,
        context="Microservices architecture handling user requests"
    )
    
    # Print detailed report
    print_anomaly_report(explained_anomalies)
    
    # Example 8: Detector Comparison
    print("\n9. Detector Comparison")
    print("-" * 80)
    
    comparison_detectors = {
        'Moving Average': ma_detector,
        'Exponential Smoothing': es_detector,
        'Isolation Forest': if_detector
    }
    
    if STATSMODELS_AVAILABLE:
        comparison_detectors['STL Decomposition'] = stl_detector
    
    results = compare_detectors(df, comparison_detectors)
    
    print("\nComparison Summary:")
    for name, anomalies in results.items():
        severities = {}
        for a in anomalies:
            severities[a.severity] = severities.get(a.severity, 0) + 1
        print(f"  {name}: {len(anomalies)} total")
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in severities:
                print(f"    {severity}: {severities[severity]}")
    
    print("\n" + "=" * 80)
    print("Examples complete!")


if __name__ == "__main__":
    main()
