"""
Chapter 10: The Alert Fatigue Problem
AI-Powered Observability: From Noise to Insight

This module provides tools for measuring and analyzing alert fatigue including:
- Alert volume and pattern analysis
- Signal-to-noise ratio calculation
- Response time and resolution tracking
- On-call impact assessment
- Alert quality scoring
- AI-powered diagnostic reports using AWS Bedrock

Author: AI-Powered Observability
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
import hashlib


@dataclass
class Alert:
    """Represents an alert event"""
    alert_id: str
    timestamp: datetime
    source: str
    severity: str  # critical, warning, info
    title: str
    description: str
    tags: List[str]
    fired_by: str  # Rule or detector that triggered it
    acknowledged: bool
    acknowledged_at: Optional[datetime]
    resolved: bool
    resolved_at: Optional[datetime]
    action_taken: str  # none, investigated, escalated, fixed
    false_positive: bool
    responder: Optional[str]


@dataclass
class OnCallShift:
    """Represents an on-call shift"""
    engineer: str
    start_time: datetime
    end_time: datetime
    alerts_received: int
    night_pages: int  # Alerts between 10pm and 6am
    actionable_alerts: int
    false_positives: int
    response_times: List[float]  # Minutes to respond
    resolution_times: List[float]  # Minutes to resolve


@dataclass
class AlertFatigueMetrics:
    """Comprehensive alert fatigue metrics"""
    period_start: datetime
    period_end: datetime
    total_alerts: int
    actionable_alerts: int
    false_positives: int
    signal_to_noise_ratio: float
    average_response_time_minutes: float
    average_resolution_time_minutes: float
    night_pages: int
    repeat_alerts: int
    unique_alert_types: int
    top_noisy_alerts: List[Tuple[str, int]]
    severity_distribution: Dict[str, int]
    fatigue_score: float  # 0-100, higher is worse


class AlertFatigueAnalyzer:
    """
    Analyzes alert patterns to measure and diagnose alert fatigue.
    Provides comprehensive metrics and AI-powered insights.
    """
    
    def __init__(self, bedrock_client):
        """Initialize with AWS Bedrock client"""
        self.bedrock_client = bedrock_client
        self.alerts_history: List[Alert] = []
        self.shifts_history: List[OnCallShift] = []
    
    def calculate_signal_to_noise(self, alerts: List[Alert]) -> float:
        """
        Calculate signal-to-noise ratio for a set of alerts.
        
        Args:
            alerts: List of alerts to analyze
            
        Returns:
            Signal-to-noise ratio (0-1), where 1 means all alerts are actionable
        """
        if not alerts:
            return 0.0
        
        actionable = sum(1 for a in alerts 
                        if a.action_taken in ['investigated', 'escalated', 'fixed'] 
                        and not a.false_positive)
        
        return actionable / len(alerts)
    
    def identify_repeat_alerts(self, 
                              alerts: List[Alert],
                              time_window_hours: int = 24) -> Dict[str, int]:
        """
        Identify alerts that fire repeatedly without resolution.
        
        Args:
            alerts: List of alerts to analyze
            time_window_hours: Time window to consider for repeats
            
        Returns:
            Dictionary mapping alert fingerprints to repeat counts
        """
        # Create fingerprint for each alert type
        alert_fingerprints = defaultdict(list)
        
        for alert in sorted(alerts, key=lambda x: x.timestamp):
            # Fingerprint based on source, title, and tags
            fingerprint = f"{alert.source}::{alert.title}::{','.join(sorted(alert.tags))}"
            alert_fingerprints[fingerprint].append(alert.timestamp)
        
        # Count repeats within time window
        repeat_counts = {}
        for fingerprint, timestamps in alert_fingerprints.items():
            if len(timestamps) < 2:
                continue
            
            # Count how many times this alert repeated
            repeat_count = 0
            for i in range(len(timestamps) - 1):
                time_diff = (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
                if time_diff <= time_window_hours:
                    repeat_count += 1
            
            if repeat_count > 0:
                repeat_counts[fingerprint] = repeat_count
        
        return repeat_counts
    
    def analyze_response_patterns(self, alerts: List[Alert]) -> Dict[str, Any]:
        """
        Analyze how quickly and effectively alerts are responded to.
        
        Args:
            alerts: List of alerts to analyze
            
        Returns:
            Dictionary with response pattern analysis
        """
        response_times = []
        resolution_times = []
        unacknowledged = 0
        unresolved = 0
        
        for alert in alerts:
            if alert.acknowledged and alert.acknowledged_at:
                response_time = (alert.acknowledged_at - alert.timestamp).total_seconds() / 60
                response_times.append(response_time)
            else:
                unacknowledged += 1
            
            if alert.resolved and alert.resolved_at:
                resolution_time = (alert.resolved_at - alert.timestamp).total_seconds() / 60
                resolution_times.append(resolution_time)
            else:
                unresolved += 1
        
        return {
            'average_response_time_minutes': np.mean(response_times) if response_times else 0,
            'median_response_time_minutes': np.median(response_times) if response_times else 0,
            'p95_response_time_minutes': np.percentile(response_times, 95) if response_times else 0,
            'average_resolution_time_minutes': np.mean(resolution_times) if resolution_times else 0,
            'median_resolution_time_minutes': np.median(resolution_times) if resolution_times else 0,
            'unacknowledged_count': unacknowledged,
            'unresolved_count': unresolved,
            'response_time_trend': self._calculate_trend(response_times) if len(response_times) > 10 else 'insufficient_data'
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction (improving, degrading, stable)"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Split into first and second half
        mid = len(values) // 2
        first_half_mean = np.mean(values[:mid])
        second_half_mean = np.mean(values[mid:])
        
        change_pct = ((second_half_mean - first_half_mean) / first_half_mean) * 100
        
        if change_pct < -10:
            return 'improving'
        elif change_pct > 10:
            return 'degrading'
        else:
            return 'stable'
    
    def calculate_night_pages(self, alerts: List[Alert]) -> int:
        """
        Count alerts that occurred during night hours (10pm-6am).
        
        Args:
            alerts: List of alerts to analyze
            
        Returns:
            Count of night alerts
        """
        night_hours = set(range(22, 24)) | set(range(0, 6))
        return sum(1 for alert in alerts if alert.timestamp.hour in night_hours)
    
    def identify_top_noisy_alerts(self, 
                                 alerts: List[Alert],
                                 top_n: int = 10) -> List[Tuple[str, int, float]]:
        """
        Identify the noisiest alert types (high volume, low action rate).
        
        Args:
            alerts: List of alerts to analyze
            top_n: Number of top noisy alerts to return
            
        Returns:
            List of (alert_type, count, false_positive_rate) tuples
        """
        # Group by alert type
        alert_types = defaultdict(lambda: {'total': 0, 'false_positives': 0})
        
        for alert in alerts:
            alert_type = f"{alert.source}::{alert.title}"
            alert_types[alert_type]['total'] += 1
            if alert.false_positive or alert.action_taken == 'none':
                alert_types[alert_type]['false_positives'] += 1
        
        # Calculate noise score (high count + high false positive rate)
        noisy_scores = []
        for alert_type, stats in alert_types.items():
            fp_rate = stats['false_positives'] / stats['total']
            noise_score = stats['total'] * fp_rate  # Count weighted by false positive rate
            noisy_scores.append((alert_type, stats['total'], fp_rate, noise_score))
        
        # Sort by noise score
        noisy_scores.sort(key=lambda x: x[3], reverse=True)
        
        return [(alert_type, count, fp_rate) 
                for alert_type, count, fp_rate, _ in noisy_scores[:top_n]]
    
    def calculate_fatigue_score(self, metrics: AlertFatigueMetrics) -> float:
        """
        Calculate an overall alert fatigue score (0-100, higher is worse).
        
        Args:
            metrics: Alert fatigue metrics
            
        Returns:
            Fatigue score
        """
        # Components of fatigue score
        components = []
        
        # Volume component: >100 alerts/day is high
        days = (metrics.period_end - metrics.period_start).days or 1
        alerts_per_day = metrics.total_alerts / days
        volume_score = min(alerts_per_day / 100 * 30, 30)  # Max 30 points
        components.append(volume_score)
        
        # Signal-to-noise component: <50% is poor
        sn_score = (1 - metrics.signal_to_noise_ratio) * 40  # Max 40 points
        components.append(sn_score)
        
        # Night pages component: >5/week is high
        weeks = days / 7
        night_pages_per_week = metrics.night_pages / weeks if weeks > 0 else metrics.night_pages
        night_score = min(night_pages_per_week / 5 * 15, 15)  # Max 15 points
        components.append(night_score)
        
        # Repeat alerts component: >20% repeat rate is high
        repeat_rate = metrics.repeat_alerts / metrics.total_alerts if metrics.total_alerts > 0 else 0
        repeat_score = min(repeat_rate * 100 / 20 * 15, 15)  # Max 15 points
        components.append(repeat_score)
        
        return sum(components)
    
    def generate_comprehensive_metrics(self, 
                                      alerts: List[Alert],
                                      period_start: datetime,
                                      period_end: datetime) -> AlertFatigueMetrics:
        """
        Generate comprehensive alert fatigue metrics for a time period.
        
        Args:
            alerts: List of alerts to analyze
            period_start: Start of analysis period
            period_end: End of analysis period
            
        Returns:
            AlertFatigueMetrics object
        """
        # Basic counts
        total_alerts = len(alerts)
        actionable = sum(1 for a in alerts 
                        if a.action_taken in ['investigated', 'escalated', 'fixed'] 
                        and not a.false_positive)
        false_positives = sum(1 for a in alerts if a.false_positive)
        
        # Signal-to-noise ratio
        signal_to_noise = self.calculate_signal_to_noise(alerts)
        
        # Response metrics
        response_patterns = self.analyze_response_patterns(alerts)
        
        # Night pages
        night_pages = self.calculate_night_pages(alerts)
        
        # Repeat alerts
        repeat_alerts_dict = self.identify_repeat_alerts(alerts)
        repeat_alerts = sum(repeat_alerts_dict.values())
        
        # Unique alert types
        unique_types = len(set(f"{a.source}::{a.title}" for a in alerts))
        
        # Top noisy alerts
        top_noisy = self.identify_top_noisy_alerts(alerts)
        
        # Severity distribution
        severity_dist = Counter(a.severity for a in alerts)
        
        # Create metrics object
        metrics = AlertFatigueMetrics(
            period_start=period_start,
            period_end=period_end,
            total_alerts=total_alerts,
            actionable_alerts=actionable,
            false_positives=false_positives,
            signal_to_noise_ratio=signal_to_noise,
            average_response_time_minutes=response_patterns['average_response_time_minutes'],
            average_resolution_time_minutes=response_patterns['average_resolution_time_minutes'],
            night_pages=night_pages,
            repeat_alerts=repeat_alerts,
            unique_alert_types=unique_types,
            top_noisy_alerts=top_noisy,
            severity_distribution=dict(severity_dist),
            fatigue_score=0  # Will be calculated below
        )
        
        # Calculate fatigue score
        metrics.fatigue_score = self.calculate_fatigue_score(metrics)
        
        return metrics
    
    def analyze_on_call_impact(self, shifts: List[OnCallShift]) -> Dict[str, Any]:
        """
        Analyze the impact of alerts on on-call engineers.
        
        Args:
            shifts: List of on-call shifts to analyze
            
        Returns:
            Dictionary with on-call impact metrics
        """
        if not shifts:
            return {}
        
        total_shifts = len(shifts)
        total_alerts = sum(s.alerts_received for s in shifts)
        total_night_pages = sum(s.night_pages for s in shifts)
        
        # Calculate per-shift statistics
        alerts_per_shift = [s.alerts_received for s in shifts]
        night_pages_per_shift = [s.night_pages for s in shifts]
        actionable_rates = [s.actionable_alerts / s.alerts_received if s.alerts_received > 0 else 0 
                           for s in shifts]
        
        # Identify problematic shifts (high alert volume or many night pages)
        problematic_shifts = [s for s in shifts 
                             if s.alerts_received > 50 or s.night_pages > 3]
        
        return {
            'total_shifts': total_shifts,
            'average_alerts_per_shift': np.mean(alerts_per_shift),
            'median_alerts_per_shift': np.median(alerts_per_shift),
            'max_alerts_per_shift': max(alerts_per_shift),
            'average_night_pages_per_shift': np.mean(night_pages_per_shift),
            'shifts_with_night_pages': sum(1 for n in night_pages_per_shift if n > 0),
            'average_actionable_rate': np.mean(actionable_rates),
            'problematic_shifts_count': len(problematic_shifts),
            'problematic_shifts_percentage': (len(problematic_shifts) / total_shifts) * 100,
            'engineers_affected': len(set(s.engineer for s in shifts))
        }
    
    def compare_time_periods(self, 
                           current_alerts: List[Alert],
                           previous_alerts: List[Alert],
                           period_name: str = "month") -> Dict[str, Any]:
        """
        Compare alert fatigue metrics between two time periods.
        
        Args:
            current_alerts: Alerts from current period
            previous_alerts: Alerts from previous period
            period_name: Name of period for reporting (week, month, quarter)
            
        Returns:
            Comparison metrics showing trends
        """
        current_sn = self.calculate_signal_to_noise(current_alerts)
        previous_sn = self.calculate_signal_to_noise(previous_alerts)
        
        current_night = self.calculate_night_pages(current_alerts)
        previous_night = self.calculate_night_pages(previous_alerts)
        
        return {
            'period': period_name,
            'alert_volume_change': len(current_alerts) - len(previous_alerts),
            'alert_volume_change_pct': ((len(current_alerts) - len(previous_alerts)) / len(previous_alerts) * 100) if previous_alerts else 0,
            'signal_to_noise_change': current_sn - previous_sn,
            'signal_to_noise_trend': 'improving' if current_sn > previous_sn else 'degrading' if current_sn < previous_sn else 'stable',
            'night_pages_change': current_night - previous_night,
            'night_pages_trend': 'improving' if current_night < previous_night else 'degrading' if current_night > previous_night else 'stable',
            'current_volume': len(current_alerts),
            'previous_volume': len(previous_alerts),
            'current_signal_to_noise': current_sn,
            'previous_signal_to_noise': previous_sn
        }
    
    def generate_diagnostic_report(self, metrics: AlertFatigueMetrics) -> str:
        """
        Generate AI-powered diagnostic report using AWS Bedrock.
        
        Args:
            metrics: Alert fatigue metrics
            
        Returns:
            Natural language diagnostic report with recommendations
        """
        prompt = f"""Analyze this alert fatigue situation and provide recommendations:

Time Period: {metrics.period_start.strftime('%Y-%m-%d')} to {metrics.period_end.strftime('%Y-%m-%d')}
Duration: {(metrics.period_end - metrics.period_start).days} days

METRICS:
- Total Alerts: {metrics.total_alerts}
- Actionable Alerts: {metrics.actionable_alerts}
- False Positives: {metrics.false_positives}
- Signal-to-Noise Ratio: {metrics.signal_to_noise_ratio:.1%}
- Average Response Time: {metrics.average_response_time_minutes:.1f} minutes
- Average Resolution Time: {metrics.average_resolution_time_minutes:.1f} minutes
- Night Pages (10pm-6am): {metrics.night_pages}
- Repeat Alerts: {metrics.repeat_alerts}
- Unique Alert Types: {metrics.unique_alert_types}
- Fatigue Score: {metrics.fatigue_score:.1f}/100 (higher is worse)

TOP NOISY ALERTS:
{chr(10).join(f"- {name}: {count} occurrences, {fp_rate:.1%} false positive rate" for name, count, fp_rate in metrics.top_noisy_alerts[:5])}

SEVERITY DISTRIBUTION:
{json.dumps(metrics.severity_distribution, indent=2)}

Provide:
1. Overall assessment of alert fatigue severity (Low/Medium/High/Critical)
2. Key problems identified
3. Specific alerts or patterns to address first (quick wins)
4. Systemic issues that need longer-term solutions
5. Impact on on-call engineer wellbeing
6. 3-5 actionable recommendations prioritized by impact

Be specific and focus on actionable improvements."""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 2000,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
    
    def generate_executive_summary(self, 
                                  metrics: AlertFatigueMetrics,
                                  on_call_impact: Dict[str, Any]) -> str:
        """
        Generate executive-level summary using AWS Bedrock.
        
        Args:
            metrics: Alert fatigue metrics
            on_call_impact: On-call impact analysis
            
        Returns:
            Executive summary suitable for leadership
        """
        prompt = f"""Create an executive summary of our alert fatigue situation:

ALERT METRICS ({(metrics.period_end - metrics.period_start).days} days):
- Total Alerts: {metrics.total_alerts}
- Signal-to-Noise Ratio: {metrics.signal_to_noise_ratio:.1%}
- Night Disruptions: {metrics.night_pages}
- Fatigue Score: {metrics.fatigue_score:.1f}/100

ON-CALL IMPACT:
- Total Shifts: {on_call_impact.get('total_shifts', 0)}
- Average Alerts per Shift: {on_call_impact.get('average_alerts_per_shift', 0):.0f}
- Problematic Shifts: {on_call_impact.get('problematic_shifts_percentage', 0):.1f}%
- Engineers Affected: {on_call_impact.get('engineers_affected', 0)}

Provide a brief (3-4 paragraphs) executive summary covering:
1. Current state and severity
2. Business impact (reliability, engineer wellbeing, operational efficiency)
3. Cost of inaction
4. High-level path forward

Use business language, not technical jargon. Focus on impacts and outcomes."""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1200,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']


class AlertQualityScorer:
    """
    Scores individual alerts for quality to identify improvement opportunities.
    """
    
    def __init__(self, bedrock_client):
        """Initialize with AWS Bedrock client"""
        self.bedrock_client = bedrock_client
    
    def score_alert_quality(self, alert: Alert, historical_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score the quality of an individual alert.
        
        Args:
            alert: Alert to score
            historical_context: Historical information about this alert type
            
        Returns:
            Quality score and breakdown
        """
        scores = {}
        
        # Actionability score (was action taken?)
        if alert.action_taken in ['investigated', 'escalated', 'fixed']:
            scores['actionability'] = 1.0
        elif alert.action_taken == 'none' and not alert.false_positive:
            scores['actionability'] = 0.5
        else:
            scores['actionability'] = 0.0
        
        # Uniqueness score (does it repeat frequently?)
        repeat_rate = historical_context.get('repeat_rate', 0)
        scores['uniqueness'] = max(0, 1.0 - repeat_rate)
        
        # Severity appropriateness (critical alerts should be actionable)
        if alert.severity == 'critical':
            if alert.action_taken in ['escalated', 'fixed']:
                scores['severity_appropriate'] = 1.0
            else:
                scores['severity_appropriate'] = 0.0
        else:
            scores['severity_appropriate'] = 0.8  # Neutral for non-critical
        
        # Response time quality (acknowledged quickly?)
        if alert.acknowledged and alert.acknowledged_at:
            response_minutes = (alert.acknowledged_at - alert.timestamp).total_seconds() / 60
            if response_minutes < 5:
                scores['response_time'] = 1.0
            elif response_minutes < 15:
                scores['response_time'] = 0.8
            elif response_minutes < 60:
                scores['response_time'] = 0.5
            else:
                scores['response_time'] = 0.2
        else:
            scores['response_time'] = 0.0
        
        # Calculate overall quality score (weighted average)
        weights = {
            'actionability': 0.4,
            'uniqueness': 0.3,
            'severity_appropriate': 0.2,
            'response_time': 0.1
        }
        
        overall_quality = sum(scores[k] * weights[k] for k in scores)
        
        return {
            'overall_quality': overall_quality,
            'component_scores': scores,
            'quality_grade': self._quality_to_grade(overall_quality),
            'needs_improvement': overall_quality < 0.6
        }
    
    def _quality_to_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def identify_low_quality_alerts(self, 
                                   alerts: List[Alert],
                                   threshold: float = 0.5) -> List[Tuple[Alert, float]]:
        """
        Identify alerts with quality scores below threshold.
        
        Args:
            alerts: List of alerts to analyze
            threshold: Quality threshold
            
        Returns:
            List of (alert, quality_score) tuples
        """
        low_quality = []
        
        # Build historical context
        alert_types = defaultdict(list)
        for alert in alerts:
            alert_type = f"{alert.source}::{alert.title}"
            alert_types[alert_type].append(alert)
        
        for alert in alerts:
            alert_type = f"{alert.source}::{alert.title}"
            same_type_alerts = alert_types[alert_type]
            
            # Calculate repeat rate for this type
            repeat_rate = len(same_type_alerts) / len(alerts)
            
            historical_context = {
                'repeat_rate': repeat_rate,
                'total_count': len(same_type_alerts)
            }
            
            quality_result = self.score_alert_quality(alert, historical_context)
            
            if quality_result['overall_quality'] < threshold:
                low_quality.append((alert, quality_result['overall_quality']))
        
        return sorted(low_quality, key=lambda x: x[1])


# Example usage and realistic data generators
def generate_sample_alerts(count: int = 200, 
                          fatigue_scenario: str = 'moderate') -> List[Alert]:
    """
    Generate realistic sample alerts with configurable fatigue scenarios.
    
    Args:
        count: Number of alerts to generate
        fatigue_scenario: 'low', 'moderate', 'high', or 'critical'
        
    Returns:
        List of sample alerts
    """
    # Configure scenarios
    scenarios = {
        'low': {
            'false_positive_rate': 0.1,
            'actionable_rate': 0.7,
            'repeat_rate': 0.1,
            'night_page_rate': 0.05
        },
        'moderate': {
            'false_positive_rate': 0.3,
            'actionable_rate': 0.4,
            'repeat_rate': 0.25,
            'night_page_rate': 0.15
        },
        'high': {
            'false_positive_rate': 0.5,
            'actionable_rate': 0.2,
            'repeat_rate': 0.4,
            'night_page_rate': 0.25
        },
        'critical': {
            'false_positive_rate': 0.7,
            'actionable_rate': 0.1,
            'repeat_rate': 0.6,
            'night_page_rate': 0.35
        }
    }
    
    config = scenarios.get(fatigue_scenario, scenarios['moderate'])
    
    alert_sources = ['kubernetes', 'prometheus', 'cloudwatch', 'datadog', 'splunk']
    alert_titles = [
        'High CPU Usage',
        'Memory Threshold Exceeded',
        'Disk Space Low',
        'API Latency High',
        'Error Rate Increased',
        'Database Connection Pool Exhausted',
        'Service Health Check Failed',
        'Queue Depth Critical',
        'Cache Hit Rate Low',
        'Network Packet Loss'
    ]
    
    severities = ['critical', 'warning', 'info']
    severity_weights = [0.2, 0.5, 0.3]
    
    actions = ['none', 'investigated', 'escalated', 'fixed']
    responders = ['alice', 'bob', 'charlie', 'diana', 'eve']
    
    alerts = []
    base_time = datetime.now() - timedelta(days=30)
    
    # Track repeated alerts
    repeat_templates = np.random.choice(alert_titles, 
                                       size=int(len(alert_titles) * config['repeat_rate']),
                                       replace=False).tolist()
    
    for i in range(count):
        # Determine timestamp
        hours_offset = np.random.uniform(0, 30 * 24)
        timestamp = base_time + timedelta(hours=hours_offset)
        
        # Night page probability
        if np.random.random() < config['night_page_rate']:
            night_hour = np.random.choice([22, 23, 0, 1, 2, 3, 4, 5])
            timestamp = timestamp.replace(hour=night_hour)
        
        # Choose alert title (favor repeats based on config)
        if repeat_templates and np.random.random() < config['repeat_rate']:
            title = np.random.choice(repeat_templates)
        else:
            title = np.random.choice(alert_titles)
        
        source = np.random.choice(alert_sources)
        severity = np.random.choice(severities, p=severity_weights)
        
        # Determine if false positive
        is_false_positive = np.random.random() < config['false_positive_rate']
        
        # Determine action taken
        if is_false_positive:
            action = 'none'
        else:
            if np.random.random() < config['actionable_rate']:
                action = np.random.choice(['investigated', 'escalated', 'fixed'], 
                                        p=[0.5, 0.3, 0.2])
            else:
                action = 'none'
        
        # Acknowledgment and resolution timing
        acknowledged = np.random.random() > 0.1  # 90% acknowledged
        acknowledged_at = None
        resolved = False
        resolved_at = None
        
        if acknowledged:
            ack_delay = np.random.exponential(10)  # Average 10 minutes
            acknowledged_at = timestamp + timedelta(minutes=ack_delay)
            
            if action != 'none':
                resolved = np.random.random() > 0.2  # 80% of actionable alerts resolved
                if resolved:
                    resolution_delay = np.random.exponential(45)  # Average 45 minutes
                    resolved_at = acknowledged_at + timedelta(minutes=resolution_delay)
        
        alerts.append(Alert(
            alert_id=f"alert_{i:04d}",
            timestamp=timestamp,
            source=source,
            severity=severity,
            title=title,
            description=f"{title} detected on {source}",
            tags=[source, severity, title.lower().replace(' ', '_')],
            fired_by=f"{source}_monitor",
            acknowledged=acknowledged,
            acknowledged_at=acknowledged_at,
            resolved=resolved,
            resolved_at=resolved_at,
            action_taken=action,
            false_positive=is_false_positive,
            responder=np.random.choice(responders) if acknowledged else None
        ))
    
    return sorted(alerts, key=lambda x: x.timestamp)


def generate_sample_shifts(alerts: List[Alert], num_engineers: int = 5) -> List[OnCallShift]:
    """Generate on-call shifts from alerts"""
    engineers = [f"engineer_{i}" for i in range(num_engineers)]
    shifts = []
    
    # Create 7-day shifts
    start_date = min(a.timestamp for a in alerts)
    end_date = max(a.timestamp for a in alerts)
    
    current = start_date
    engineer_idx = 0
    
    while current < end_date:
        shift_end = current + timedelta(days=7)
        engineer = engineers[engineer_idx % len(engineers)]
        
        # Get alerts during this shift
        shift_alerts = [a for a in alerts if current <= a.timestamp < shift_end]
        
        actionable = sum(1 for a in shift_alerts 
                        if a.action_taken in ['investigated', 'escalated', 'fixed'])
        false_positives = sum(1 for a in shift_alerts if a.false_positive)
        
        night_hours = set(range(22, 24)) | set(range(0, 6))
        night_pages = sum(1 for a in shift_alerts if a.timestamp.hour in night_hours)
        
        response_times = []
        resolution_times = []
        for a in shift_alerts:
            if a.acknowledged and a.acknowledged_at:
                response_times.append((a.acknowledged_at - a.timestamp).total_seconds() / 60)
            if a.resolved and a.resolved_at:
                resolution_times.append((a.resolved_at - a.timestamp).total_seconds() / 60)
        
        shifts.append(OnCallShift(
            engineer=engineer,
            start_time=current,
            end_time=shift_end,
            alerts_received=len(shift_alerts),
            night_pages=night_pages,
            actionable_alerts=actionable,
            false_positives=false_positives,
            response_times=response_times,
            resolution_times=resolution_times
        ))
        
        current = shift_end
        engineer_idx += 1
    
    return shifts


def main():
    """
    Demonstrate alert fatigue analysis with realistic scenarios.
    """
    print("=" * 80)
    print("Chapter 10: The Alert Fatigue Problem - Analysis Tool")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    print("Initializing alert fatigue analyzer...")
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    analyzer = AlertFatigueAnalyzer(bedrock_client)
    quality_scorer = AlertQualityScorer(bedrock_client)
    print()
    
    # Generate alerts for different scenarios
    print("Analyzing Alert Fatigue Scenarios")
    print("-" * 80)
    
    scenarios = ['low', 'moderate', 'high']
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario.upper()} Alert Fatigue")
        print('='*80)
        print()
        
        # Generate sample data
        alerts = generate_sample_alerts(count=300, fatigue_scenario=scenario)
        shifts = generate_sample_shifts(alerts, num_engineers=5)
        
        period_start = min(a.timestamp for a in alerts)
        period_end = max(a.timestamp for a in alerts)
        
        # Generate metrics
        print("Calculating comprehensive metrics...")
        metrics = analyzer.generate_comprehensive_metrics(alerts, period_start, period_end)
        
        # Print summary
        print(f"\nMETRICS SUMMARY ({(period_end - period_start).days} days):")
        print("-" * 80)
        print(f"Total Alerts: {metrics.total_alerts}")
        print(f"Actionable Alerts: {metrics.actionable_alerts}")
        print(f"False Positives: {metrics.false_positives}")
        print(f"Signal-to-Noise Ratio: {metrics.signal_to_noise_ratio:.1%}")
        print(f"Average Response Time: {metrics.average_response_time_minutes:.1f} minutes")
        print(f"Average Resolution Time: {metrics.average_resolution_time_minutes:.1f} minutes")
        print(f"Night Pages: {metrics.night_pages}")
        print(f"Repeat Alerts: {metrics.repeat_alerts}")
        print(f"Unique Alert Types: {metrics.unique_alert_types}")
        print(f"\nFATIGUE SCORE: {metrics.fatigue_score:.1f}/100")
        
        if metrics.fatigue_score < 30:
            print("Assessment: LOW fatigue - System is healthy")
        elif metrics.fatigue_score < 50:
            print("Assessment: MODERATE fatigue - Improvement needed")
        elif metrics.fatigue_score < 70:
            print("Assessment: HIGH fatigue - Urgent attention required")
        else:
            print("Assessment: CRITICAL fatigue - Immediate action needed")
        
        print(f"\nTop 5 Noisy Alerts:")
        for i, (alert_type, count, fp_rate) in enumerate(metrics.top_noisy_alerts[:5], 1):
            print(f"{i}. {alert_type}")
            print(f"   Count: {count}, False Positive Rate: {fp_rate:.1%}")
        
        # On-call impact
        print(f"\nON-CALL IMPACT:")
        print("-" * 80)
        on_call_impact = analyzer.analyze_on_call_impact(shifts)
        print(f"Total Shifts: {on_call_impact['total_shifts']}")
        print(f"Average Alerts per Shift: {on_call_impact['average_alerts_per_shift']:.0f}")
        print(f"Problematic Shifts: {on_call_impact['problematic_shifts_percentage']:.1f}%")
        print(f"Shifts with Night Pages: {on_call_impact['shifts_with_night_pages']}")
        print(f"Average Actionable Rate: {on_call_impact['average_actionable_rate']:.1%}")
        
        # Generate AI diagnostic report
        print(f"\n{'='*80}")
        print("AI-POWERED DIAGNOSTIC REPORT")
        print('='*80)
        print()
        diagnostic = analyzer.generate_diagnostic_report(metrics)
        print(diagnostic)
        
        # Executive summary (only for moderate and high)
        if scenario in ['moderate', 'high']:
            print(f"\n{'='*80}")
            print("EXECUTIVE SUMMARY")
            print('='*80)
            print()
            exec_summary = analyzer.generate_executive_summary(metrics, on_call_impact)
            print(exec_summary)
        
        print("\n")
    
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
