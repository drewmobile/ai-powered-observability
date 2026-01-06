"""
Chapter 12: Predictive Alerting
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready predictive alerting including:
- Time series forecasting for resource exhaustion
- Trend extrapolation for capacity prediction
- Pre-failure state classification
- Anomaly-based early warning systems
- Capacity modeling and threshold prediction
- Lead time calculation and confidence scoring
- AI-powered prediction explanations with AWS Bedrock

Author: AI-Powered Observability
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MetricDataPoint:
    """Represents a single metric data point"""
    timestamp: datetime
    value: float
    metric_name: str
    entity: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictiveAlert:
    """Represents a predictive alert"""
    alert_id: str
    timestamp: datetime
    metric_name: str
    entity: str
    prediction_type: str  # resource_exhaustion, capacity_threshold, pre_failure_state
    predicted_event: str
    predicted_time: datetime
    lead_time_minutes: float
    confidence: float
    severity: str
    evidence: Dict[str, Any]
    recommended_actions: List[str]
    explanation: str


@dataclass
class SystemState:
    """Represents current system state for classification"""
    timestamp: datetime
    entity: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    error_rate: float
    latency_p95: float
    request_rate: float
    active_connections: int


class ResourceExhaustionPredictor:
    """
    Predicts resource exhaustion events using time series forecasting
    and trend extrapolation.
    """
    
    def __init__(self, bedrock_client):
        """Initialize predictor with AWS Bedrock client"""
        self.bedrock_client = bedrock_client
        self.metric_history: Dict[str, List[MetricDataPoint]] = defaultdict(list)
    
    def add_metric_data(self, data_point: MetricDataPoint):
        """Add metric data point to history"""
        key = f"{data_point.entity}::{data_point.metric_name}"
        self.metric_history[key].append(data_point)
        
        # Keep only recent history (e.g., last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.metric_history[key] = [
            dp for dp in self.metric_history[key]
            if dp.timestamp > cutoff
        ]
    
    def predict_resource_exhaustion(self,
                                   entity: str,
                                   metric_name: str,
                                   threshold: float = 95.0,
                                   min_lead_time_hours: float = 1.0) -> Optional[PredictiveAlert]:
        """
        Predict when a resource will exceed threshold.
        
        Args:
            entity: Entity being monitored
            metric_name: Metric to predict (e.g., 'memory_usage_percent')
            threshold: Threshold value to predict crossing
            min_lead_time_hours: Minimum lead time for actionable alert
            
        Returns:
            PredictiveAlert if exhaustion is predicted, None otherwise
        """
        key = f"{entity}::{metric_name}"
        
        if key not in self.metric_history or len(self.metric_history[key]) < 10:
            return None  # Insufficient data
        
        data_points = self.metric_history[key]
        
        # Extract time series
        timestamps = [dp.timestamp for dp in data_points]
        values = [dp.value for dp in data_points]
        
        # Fit linear trend
        X = np.array([(ts - timestamps[0]).total_seconds() / 3600 
                     for ts in timestamps]).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Current value and trend
        current_value = values[-1]
        trend_per_hour = model.coef_[0]
        
        # Only alert if trending upward
        if trend_per_hour <= 0:
            return None
        
        # Calculate time to threshold
        time_to_threshold_hours = (threshold - current_value) / trend_per_hour
        
        # Check if prediction is within actionable window
        if time_to_threshold_hours < min_lead_time_hours:
            return None  # Too soon to act
        
        if time_to_threshold_hours > 168:  # 1 week
            return None  # Too far out to be reliable
        
        # Calculate prediction confidence
        # Based on trend consistency (R-squared)
        confidence = self._calculate_trend_confidence(X, y, model)
        
        if confidence < 0.5:
            return None  # Low confidence prediction
        
        # Generate predicted time
        predicted_time = datetime.now() + timedelta(hours=time_to_threshold_hours)
        
        # Determine severity based on lead time
        if time_to_threshold_hours < 4:
            severity = 'critical'
        elif time_to_threshold_hours < 24:
            severity = 'warning'
        else:
            severity = 'info'
        
        # Generate evidence
        evidence = {
            'current_value': current_value,
            'threshold': threshold,
            'trend_per_hour': trend_per_hour,
            'time_to_threshold_hours': time_to_threshold_hours,
            'historical_data_points': len(data_points),
            'r_squared': confidence
        }
        
        # Generate recommended actions
        recommended_actions = self._generate_recommendations(
            metric_name,
            trend_per_hour,
            time_to_threshold_hours
        )
        
        # Generate AI explanation
        explanation = self._generate_prediction_explanation(
            entity,
            metric_name,
            evidence,
            recommended_actions
        )
        
        alert_id = f"pred_{entity}_{metric_name}_{datetime.now().timestamp()}"
        
        return PredictiveAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            metric_name=metric_name,
            entity=entity,
            prediction_type='resource_exhaustion',
            predicted_event=f"{metric_name} will exceed {threshold}%",
            predicted_time=predicted_time,
            lead_time_minutes=time_to_threshold_hours * 60,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            recommended_actions=recommended_actions,
            explanation=explanation
        )
    
    def _calculate_trend_confidence(self, X, y, model) -> float:
        """Calculate confidence in trend prediction using R-squared"""
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    
    def _generate_recommendations(self,
                                 metric_name: str,
                                 trend: float,
                                 hours_remaining: float) -> List[str]:
        """Generate recommended actions based on prediction"""
        recommendations = []
        
        if 'memory' in metric_name.lower():
            recommendations.append("Investigate memory usage patterns for potential leaks")
            recommendations.append("Consider increasing memory limits or scaling horizontally")
            if hours_remaining > 24:
                recommendations.append("Review recent code changes for memory-intensive operations")
            else:
                recommendations.append("Prepare to restart service if leak is confirmed")
        
        elif 'disk' in metric_name.lower():
            recommendations.append("Identify and clean up unnecessary files")
            recommendations.append("Review log rotation and retention policies")
            recommendations.append("Consider expanding disk capacity")
            if hours_remaining < 12:
                recommendations.append("Urgent: Implement emergency cleanup procedures")
        
        elif 'cpu' in metric_name.lower():
            recommendations.append("Analyze CPU hotspots with profiling tools")
            recommendations.append("Consider horizontal scaling to distribute load")
            recommendations.append("Review recent deployments for performance regressions")
        
        elif 'connection' in metric_name.lower():
            recommendations.append("Review connection pool configuration")
            recommendations.append("Investigate connection leaks")
            recommendations.append("Consider increasing connection pool limits")
        
        else:
            recommendations.append("Monitor the situation closely")
            recommendations.append("Review recent system changes")
        
        return recommendations
    
    def _generate_prediction_explanation(self,
                                        entity: str,
                                        metric_name: str,
                                        evidence: Dict[str, Any],
                                        recommendations: List[str]) -> str:
        """
        Generate natural language explanation using AWS Bedrock.
        
        Args:
            entity: Entity name
            metric_name: Metric being predicted
            evidence: Evidence dictionary
            recommendations: List of recommendations
            
        Returns:
            Natural language explanation
        """
        prompt = f"""Explain this predictive alert to an on-call engineer:

Entity: {entity}
Metric: {metric_name}
Current Value: {evidence['current_value']:.1f}%
Threshold: {evidence['threshold']:.1f}%
Growth Rate: {evidence['trend_per_hour']:.2f}% per hour
Predicted Exhaustion: {evidence['time_to_threshold_hours']:.1f} hours
Confidence: {evidence['r_squared']:.0%}

Recommended Actions:
{chr(10).join(f"- {rec}" for rec in recommendations)}

Provide a brief (2-3 sentences) explanation:
1. What's happening
2. Why it matters
3. What should be done

Be specific and actionable."""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 300,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']


class CapacityPredictor:
    """
    Predicts capacity thresholds using demand forecasting and capacity modeling.
    """
    
    def __init__(self, bedrock_client):
        """Initialize capacity predictor"""
        self.bedrock_client = bedrock_client
        self.traffic_history: Dict[str, List[MetricDataPoint]] = defaultdict(list)
        self.capacity_limits: Dict[str, float] = {}
    
    def set_capacity_limit(self, entity: str, limit: float):
        """Set known capacity limit for an entity"""
        self.capacity_limits[entity] = limit
    
    def add_traffic_data(self, data_point: MetricDataPoint):
        """Add traffic/load data point"""
        key = data_point.entity
        self.traffic_history[key].append(data_point)
        
        # Keep last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        self.traffic_history[key] = [
            dp for dp in self.traffic_history[key]
            if dp.timestamp > cutoff
        ]
    
    def predict_capacity_exhaustion(self,
                                   entity: str,
                                   forecast_hours: int = 168) -> Optional[PredictiveAlert]:
        """
        Predict when traffic will exceed capacity.
        
        Args:
            entity: Entity to predict for
            forecast_hours: Hours to forecast ahead
            
        Returns:
            PredictiveAlert if capacity exhaustion predicted
        """
        if entity not in self.capacity_limits:
            return None  # No capacity limit defined
        
        if entity not in self.traffic_history or len(self.traffic_history[entity]) < 24:
            return None  # Insufficient data
        
        capacity = self.capacity_limits[entity]
        data_points = self.traffic_history[entity]
        
        # Extract time series
        timestamps = [dp.timestamp for dp in data_points]
        values = [dp.value for dp in data_points]
        
        # Simple forecast: extrapolate recent trend
        recent_points = data_points[-24:]  # Last 24 data points
        recent_times = [(dp.timestamp - recent_points[0].timestamp).total_seconds() / 3600
                       for dp in recent_points]
        recent_values = [dp.value for dp in recent_points]
        
        # Fit trend
        X = np.array(recent_times).reshape(-1, 1)
        y = np.array(recent_values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        current_value = values[-1]
        growth_rate = model.coef_[0]
        
        # Check if we'll exceed capacity
        if growth_rate <= 0:
            return None  # Not growing
        
        hours_to_capacity = (capacity - current_value) / growth_rate
        
        if hours_to_capacity < 1 or hours_to_capacity > forecast_hours:
            return None  # Outside actionable window
        
        # Calculate confidence
        confidence = self._calculate_trend_confidence(X, y, model)
        
        if confidence < 0.6:
            return None
        
        predicted_time = datetime.now() + timedelta(hours=hours_to_capacity)
        
        # Determine severity
        if hours_to_capacity < 4:
            severity = 'critical'
        elif hours_to_capacity < 24:
            severity = 'warning'
        else:
            severity = 'info'
        
        evidence = {
            'current_traffic': current_value,
            'capacity_limit': capacity,
            'utilization_percent': (current_value / capacity) * 100,
            'growth_rate_per_hour': growth_rate,
            'hours_to_capacity': hours_to_capacity,
            'confidence': confidence
        }
        
        recommended_actions = [
            "Review auto-scaling configuration",
            "Prepare to scale capacity before predicted exhaustion",
            "Analyze traffic patterns for potential optimization",
            "Consider load balancing adjustments"
        ]
        
        explanation = self._generate_capacity_explanation(entity, evidence)
        
        return PredictiveAlert(
            alert_id=f"capacity_{entity}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            metric_name='traffic_volume',
            entity=entity,
            prediction_type='capacity_threshold',
            predicted_event=f"Traffic will exceed capacity limit of {capacity}",
            predicted_time=predicted_time,
            lead_time_minutes=hours_to_capacity * 60,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            recommended_actions=recommended_actions,
            explanation=explanation
        )
    
    def _calculate_trend_confidence(self, X, y, model) -> float:
        """Calculate R-squared for trend confidence"""
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return max(0.0, 1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    def _generate_capacity_explanation(self,
                                     entity: str,
                                     evidence: Dict[str, Any]) -> str:
        """Generate capacity prediction explanation"""
        prompt = f"""Explain this capacity prediction alert:

Entity: {entity}
Current Traffic: {evidence['current_traffic']:.0f} requests/sec
Capacity Limit: {evidence['capacity_limit']:.0f} requests/sec
Current Utilization: {evidence['utilization_percent']:.1f}%
Growth Rate: {evidence['growth_rate_per_hour']:.1f} req/sec per hour
Time to Capacity: {evidence['hours_to_capacity']:.1f} hours

Explain in 2-3 sentences:
1. Current capacity situation
2. When action is needed
3. What should be done

Be specific about timing and urgency."""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 250,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']


class PreFailureStateClassifier:
    """
    Classifies system states as normal or pre-failure using ML.
    Learns patterns from historical failures.
    """
    
    def __init__(self, bedrock_client):
        """Initialize classifier"""
        self.bedrock_client = bedrock_client
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = [
            'cpu_usage', 'memory_usage', 'disk_usage',
            'error_rate', 'latency_p95', 'request_rate', 'active_connections'
        ]
    
    def train(self, historical_states: List[Tuple[SystemState, bool]]):
        """
        Train classifier on historical system states.
        
        Args:
            historical_states: List of (SystemState, is_pre_failure) tuples
        """
        if len(historical_states) < 20:
            return  # Need minimum training data
        
        # Extract features
        X = []
        y = []
        
        for state, is_pre_failure in historical_states:
            features = [
                state.cpu_usage,
                state.memory_usage,
                state.disk_usage,
                state.error_rate,
                state.latency_p95,
                state.request_rate,
                state.active_connections
            ]
            X.append(features)
            y.append(1 if is_pre_failure else 0)
        
        # Train model
        self.classifier.fit(np.array(X), np.array(y))
        self.is_trained = True
    
    def predict_failure_risk(self, state: SystemState) -> Optional[PredictiveAlert]:
        """
        Predict if current state indicates impending failure.
        
        Args:
            state: Current system state
            
        Returns:
            PredictiveAlert if high failure risk detected
        """
        if not self.is_trained:
            return None
        
        # Extract features
        features = np.array([[
            state.cpu_usage,
            state.memory_usage,
            state.disk_usage,
            state.error_rate,
            state.latency_p95,
            state.request_rate,
            state.active_connections
        ]])
        
        # Get prediction and probability
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        failure_probability = probabilities[1] if len(probabilities) > 1 else 0
        
        # Only alert on high-confidence pre-failure predictions
        if prediction == 1 and failure_probability >= 0.7:
            # Get feature importances
            importances = self.classifier.feature_importances_
            top_features = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Determine severity based on probability
            if failure_probability >= 0.9:
                severity = 'critical'
            elif failure_probability >= 0.8:
                severity = 'warning'
            else:
                severity = 'info'
            
            evidence = {
                'failure_probability': failure_probability,
                'cpu_usage': state.cpu_usage,
                'memory_usage': state.memory_usage,
                'disk_usage': state.disk_usage,
                'error_rate': state.error_rate,
                'latency_p95': state.latency_p95,
                'top_risk_factors': [
                    f"{name}: {value:.1f}" for name, value in top_features
                ]
            }
            
            recommended_actions = self._generate_state_recommendations(
                state,
                top_features
            )
            
            explanation = self._generate_state_explanation(
                state,
                failure_probability,
                top_features
            )
            
            # Estimate lead time based on historical patterns (simplified)
            estimated_lead_time_hours = 2.0  # Could be learned from data
            predicted_time = datetime.now() + timedelta(hours=estimated_lead_time_hours)
            
            return PredictiveAlert(
                alert_id=f"preFailure_{state.entity}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                metric_name='system_state',
                entity=state.entity,
                prediction_type='pre_failure_state',
                predicted_event='System failure or degradation',
                predicted_time=predicted_time,
                lead_time_minutes=estimated_lead_time_hours * 60,
                confidence=failure_probability,
                severity=severity,
                evidence=evidence,
                recommended_actions=recommended_actions,
                explanation=explanation
            )
        
        return None
    
    def _generate_state_recommendations(self,
                                       state: SystemState,
                                       top_features: List[Tuple[str, float]]) -> List[str]:
        """Generate recommendations based on risky features"""
        recommendations = []
        
        for feature_name, _ in top_features:
            if 'memory' in feature_name and state.memory_usage > 80:
                recommendations.append("Investigate memory usage - potential leak detected")
            elif 'cpu' in feature_name and state.cpu_usage > 80:
                recommendations.append("High CPU usage - review recent changes")
            elif 'error_rate' in feature_name and state.error_rate > 1:
                recommendations.append("Elevated error rate - investigate error patterns")
            elif 'latency' in feature_name and state.latency_p95 > 1000:
                recommendations.append("High latency detected - analyze slow queries/requests")
        
        if not recommendations:
            recommendations.append("Monitor system closely for developing issues")
        
        recommendations.append("Review recent deployments and changes")
        recommendations.append("Prepare incident response team")
        
        return recommendations
    
    def _generate_state_explanation(self,
                                   state: SystemState,
                                   probability: float,
                                   top_features: List[Tuple[str, float]]) -> str:
        """Generate explanation of pre-failure state"""
        prompt = f"""Explain this pre-failure prediction:

Entity: {state.entity}
Failure Probability: {probability:.0%}

Current System State:
- CPU Usage: {state.cpu_usage:.1f}%
- Memory Usage: {state.memory_usage:.1f}%
- Disk Usage: {state.disk_usage:.1f}%
- Error Rate: {state.error_rate:.2f}%
- P95 Latency: {state.latency_p95:.0f}ms

Top Risk Factors:
{chr(10).join(f"- {name}" for name, _ in top_features)}

Explain in 2-3 sentences why this state indicates potential failure and what the team should focus on."""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 250,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']


class PredictiveAlertingPipeline:
    """
    Production pipeline orchestrating all predictive alerting approaches.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize predictive alerting pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.resource_predictor = ResourceExhaustionPredictor(self.bedrock_client)
        self.capacity_predictor = CapacityPredictor(self.bedrock_client)
        self.state_classifier = PreFailureStateClassifier(self.bedrock_client)
        
        self.active_predictions: Dict[str, PredictiveAlert] = {}
        self.prediction_history: List[PredictiveAlert] = []
    
    def configure_capacity_limit(self, entity: str, limit: float):
        """Configure capacity limit for an entity"""
        self.capacity_predictor.set_capacity_limit(entity, limit)
    
    def train_failure_classifier(self, historical_states: List[Tuple[SystemState, bool]]):
        """Train the pre-failure state classifier"""
        print(f"Training failure classifier with {len(historical_states)} historical states...")
        self.state_classifier.train(historical_states)
        print("Training complete")
    
    def process_metrics(self, data_points: List[MetricDataPoint]) -> List[PredictiveAlert]:
        """
        Process metric data and generate predictive alerts.
        
        Args:
            data_points: List of metric data points
            
        Returns:
            List of generated predictive alerts
        """
        new_alerts = []
        
        for dp in data_points:
            # Add to history
            self.resource_predictor.add_metric_data(dp)
            
            # Check for resource exhaustion predictions
            if any(keyword in dp.metric_name.lower() 
                  for keyword in ['memory', 'disk', 'cpu', 'connection']):
                
                alert = self.resource_predictor.predict_resource_exhaustion(
                    dp.entity,
                    dp.metric_name,
                    threshold=95.0
                )
                
                if alert:
                    self._add_alert(alert)
                    new_alerts.append(alert)
        
        return new_alerts
    
    def process_traffic_data(self, data_points: List[MetricDataPoint]) -> List[PredictiveAlert]:
        """
        Process traffic/load data for capacity predictions.
        
        Args:
            data_points: Traffic data points
            
        Returns:
            List of capacity prediction alerts
        """
        new_alerts = []
        
        for dp in data_points:
            self.capacity_predictor.add_traffic_data(dp)
            
            alert = self.capacity_predictor.predict_capacity_exhaustion(dp.entity)
            
            if alert:
                self._add_alert(alert)
                new_alerts.append(alert)
        
        return new_alerts
    
    def process_system_state(self, state: SystemState) -> Optional[PredictiveAlert]:
        """
        Process current system state for pre-failure detection.
        
        Args:
            state: Current system state
            
        Returns:
            PredictiveAlert if failure risk detected
        """
        alert = self.state_classifier.predict_failure_risk(state)
        
        if alert:
            self._add_alert(alert)
        
        return alert
    
    def _add_alert(self, alert: PredictiveAlert):
        """Add alert to active predictions"""
        # Deduplicate by entity and prediction type
        key = f"{alert.entity}::{alert.prediction_type}"
        
        # Only keep if it's a new prediction or updated prediction
        if key not in self.active_predictions:
            self.active_predictions[key] = alert
            self.prediction_history.append(alert)
        else:
            # Update if lead time is shorter or confidence is higher
            existing = self.active_predictions[key]
            if (alert.lead_time_minutes < existing.lead_time_minutes or
                alert.confidence > existing.confidence):
                self.active_predictions[key] = alert
                self.prediction_history.append(alert)
    
    def get_active_predictions(self,
                              min_severity: str = 'info') -> List[PredictiveAlert]:
        """
        Get all active predictions above minimum severity.
        
        Args:
            min_severity: Minimum severity level
            
        Returns:
            List of active predictions
        """
        severity_order = {'info': 0, 'warning': 1, 'critical': 2}
        min_level = severity_order.get(min_severity, 0)
        
        return [
            alert for alert in self.active_predictions.values()
            if severity_order.get(alert.severity, 0) >= min_level
        ]
    
    def resolve_prediction(self, alert_id: str, outcome: str):
        """
        Record outcome of a prediction for accuracy tracking.
        
        Args:
            alert_id: Alert identifier
            outcome: 'prevented', 'occurred', 'false_positive'
        """
        # Find and mark the prediction
        for alert in self.prediction_history:
            if alert.alert_id == alert_id:
                alert.evidence['outcome'] = outcome
                alert.evidence['outcome_time'] = datetime.now()
                break
        
        # Remove from active predictions
        keys_to_remove = [
            k for k, v in self.active_predictions.items()
            if v.alert_id == alert_id
        ]
        for key in keys_to_remove:
            del self.active_predictions[key]
    
    def get_prediction_accuracy_stats(self) -> Dict[str, Any]:
        """Calculate prediction accuracy statistics"""
        outcomes = [
            alert.evidence.get('outcome')
            for alert in self.prediction_history
            if 'outcome' in alert.evidence
        ]
        
        if not outcomes:
            return {
                'total_predictions': len(self.prediction_history),
                'resolved_predictions': 0,
                'accuracy_unavailable': True
            }
        
        outcome_counts = Counter(outcomes)
        
        return {
            'total_predictions': len(self.prediction_history),
            'resolved_predictions': len(outcomes),
            'prevented_count': outcome_counts.get('prevented', 0),
            'occurred_count': outcome_counts.get('occurred', 0),
            'false_positive_count': outcome_counts.get('false_positive', 0),
            'prevention_rate': outcome_counts.get('prevented', 0) / len(outcomes) if outcomes else 0,
            'false_positive_rate': outcome_counts.get('false_positive', 0) / len(outcomes) if outcomes else 0
        }


# Example usage and realistic data generators
def generate_sample_metrics(scenario: str = 'memory_leak') -> List[MetricDataPoint]:
    """Generate realistic sample metrics"""
    data_points = []
    base_time = datetime.now() - timedelta(hours=48)
    
    if scenario == 'memory_leak':
        # Simulate gradual memory increase (leak)
        for i in range(96):  # 48 hours of 30-min intervals
            timestamp = base_time + timedelta(minutes=30*i)
            # Memory grows linearly with some noise
            memory = 45 + (i * 0.5) + np.random.normal(0, 2)
            memory = min(memory, 100)
            
            data_points.append(MetricDataPoint(
                timestamp=timestamp,
                value=memory,
                metric_name='memory_usage_percent',
                entity='payment-service-pod-1'
            ))
    
    elif scenario == 'disk_filling':
        # Simulate disk filling up
        for i in range(168):  # 7 days of hourly data
            timestamp = base_time + timedelta(hours=i-120)  # Start 5 days ago
            disk = 70 + (i * 0.15) + np.random.normal(0, 1)
            disk = min(disk, 100)
            
            data_points.append(MetricDataPoint(
                timestamp=timestamp,
                value=disk,
                metric_name='disk_usage_percent',
                entity='database-primary'
            ))
    
    return data_points


def generate_sample_traffic(scenario: str = 'growth') -> List[MetricDataPoint]:
    """Generate sample traffic data"""
    data_points = []
    base_time = datetime.now() - timedelta(days=7)
    
    if scenario == 'growth':
        # Simulate traffic growth
        for i in range(168):  # 7 days hourly
            timestamp = base_time + timedelta(hours=i)
            # Base traffic with daily pattern + growth trend
            hour_of_day = timestamp.hour
            daily_pattern = 800 + 400 * np.sin((hour_of_day - 6) * np.pi / 12)
            growth = i * 5  # Growing 5 req/sec per hour
            traffic = daily_pattern + growth + np.random.normal(0, 50)
            
            data_points.append(MetricDataPoint(
                timestamp=timestamp,
                value=max(0, traffic),
                metric_name='requests_per_second',
                entity='api-gateway'
            ))
    
    return data_points


def generate_sample_states() -> List[Tuple[SystemState, bool]]:
    """Generate sample system states for training"""
    states = []
    
    # Normal states
    for i in range(40):
        state = SystemState(
            timestamp=datetime.now() - timedelta(hours=i),
            entity='web-service',
            cpu_usage=np.random.uniform(30, 70),
            memory_usage=np.random.uniform(40, 70),
            disk_usage=np.random.uniform(50, 75),
            error_rate=np.random.uniform(0, 0.5),
            latency_p95=np.random.uniform(100, 300),
            request_rate=np.random.uniform(800, 1200),
            active_connections=np.random.randint(50, 150)
        )
        states.append((state, False))
    
    # Pre-failure states
    for i in range(20):
        state = SystemState(
            timestamp=datetime.now() - timedelta(hours=i),
            entity='web-service',
            cpu_usage=np.random.uniform(80, 95),
            memory_usage=np.random.uniform(85, 98),
            disk_usage=np.random.uniform(75, 90),
            error_rate=np.random.uniform(2, 8),
            latency_p95=np.random.uniform(800, 2000),
            request_rate=np.random.uniform(600, 1000),
            active_connections=np.random.randint(200, 400)
        )
        states.append((state, True))
    
    return states


def main():
    """
    Demonstrate predictive alerting pipeline with realistic scenarios.
    """
    print("=" * 80)
    print("Chapter 12: Predictive Alerting")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing predictive alerting pipeline...")
    pipeline = PredictiveAlertingPipeline(aws_region='us-east-1')
    print()
    
    # Configure capacity limits
    pipeline.configure_capacity_limit('api-gateway', 2000)  # 2000 req/sec capacity
    print("Configured capacity limits")
    print()
    
    # Train failure classifier
    print("Training pre-failure state classifier...")
    historical_states = generate_sample_states()
    pipeline.train_failure_classifier(historical_states)
    print()
    
    # Scenario 1: Memory Leak Detection
    print("=" * 80)
    print("SCENARIO 1: Predicting Memory Exhaustion")
    print("=" * 80)
    print()
    
    memory_metrics = generate_sample_metrics(scenario='memory_leak')
    print(f"Processing {len(memory_metrics)} memory usage data points...")
    
    alerts = pipeline.process_metrics(memory_metrics)
    
    if alerts:
        for alert in alerts:
            print(f"\n🔮 PREDICTIVE ALERT: {alert.alert_id}")
            print(f"   Severity: {alert.severity.upper()}")
            print(f"   Entity: {alert.entity}")
            print(f"   Prediction: {alert.predicted_event}")
            print(f"   Predicted Time: {alert.predicted_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Lead Time: {alert.lead_time_minutes / 60:.1f} hours")
            print(f"   Confidence: {alert.confidence:.0%}")
            print(f"\n   Evidence:")
            for key, value in alert.evidence.items():
                if isinstance(value, float):
                    print(f"     - {key}: {value:.2f}")
                else:
                    print(f"     - {key}: {value}")
            print(f"\n   Recommended Actions:")
            for action in alert.recommended_actions:
                print(f"     • {action}")
            print(f"\n   Explanation:")
            print(f"   {alert.explanation}")
    else:
        print("No predictive alerts generated")
    
    # Scenario 2: Capacity Exhaustion
    print("\n" + "=" * 80)
    print("SCENARIO 2: Predicting Capacity Exhaustion")
    print("=" * 80)
    print()
    
    traffic_data = generate_sample_traffic(scenario='growth')
    print(f"Processing {len(traffic_data)} traffic data points...")
    
    capacity_alerts = pipeline.process_traffic_data(traffic_data)
    
    if capacity_alerts:
        for alert in capacity_alerts:
            print(f"\n🔮 CAPACITY PREDICTION: {alert.alert_id}")
            print(f"   Severity: {alert.severity.upper()}")
            print(f"   Entity: {alert.entity}")
            print(f"   Prediction: {alert.predicted_event}")
            print(f"   Predicted Time: {alert.predicted_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Lead Time: {alert.lead_time_minutes / 60:.1f} hours")
            print(f"   Confidence: {alert.confidence:.0%}")
            print(f"\n   Explanation:")
            print(f"   {alert.explanation}")
    
    # Scenario 3: Pre-Failure State Detection
    print("\n" + "=" * 80)
    print("SCENARIO 3: Pre-Failure State Detection")
    print("=" * 80)
    print()
    
    # Create a high-risk current state
    risky_state = SystemState(
        timestamp=datetime.now(),
        entity='web-service',
        cpu_usage=89.5,
        memory_usage=92.3,
        disk_usage=78.5,
        error_rate=4.2,
        latency_p95=1450,
        request_rate=750,
        active_connections=325
    )
    
    print("Analyzing current system state...")
    print(f"  CPU: {risky_state.cpu_usage:.1f}%")
    print(f"  Memory: {risky_state.memory_usage:.1f}%")
    print(f"  Error Rate: {risky_state.error_rate:.1f}%")
    print(f"  P95 Latency: {risky_state.latency_p95:.0f}ms")
    print()
    
    state_alert = pipeline.process_system_state(risky_state)
    
    if state_alert:
        print(f"🔮 PRE-FAILURE STATE DETECTED: {state_alert.alert_id}")
        print(f"   Severity: {state_alert.severity.upper()}")
        print(f"   Failure Probability: {state_alert.confidence:.0%}")
        print(f"   Estimated Time to Failure: {state_alert.lead_time_minutes / 60:.1f} hours")
        print(f"\n   Top Risk Factors:")
        for factor in state_alert.evidence['top_risk_factors']:
            print(f"     • {factor}")
        print(f"\n   Explanation:")
        print(f"   {state_alert.explanation}")
    
    # Show summary
    print("\n" + "=" * 80)
    print("ACTIVE PREDICTIONS SUMMARY")
    print("=" * 80)
    print()
    
    active = pipeline.get_active_predictions()
    print(f"Total Active Predictions: {len(active)}")
    
    for alert in active:
        print(f"\n  [{alert.severity.upper()}] {alert.entity}")
        print(f"    Prediction: {alert.predicted_event}")
        print(f"    Lead Time: {alert.lead_time_minutes / 60:.1f} hours")
        print(f"    Confidence: {alert.confidence:.0%}")
    
    # Show statistics
    stats = pipeline.get_prediction_accuracy_stats()
    print(f"\nPrediction Statistics:")
    print(f"  Total Predictions Made: {stats['total_predictions']}")
    print(f"  Resolved Predictions: {stats['resolved_predictions']}")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
