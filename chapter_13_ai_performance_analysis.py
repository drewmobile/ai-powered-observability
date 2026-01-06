"""
Chapter 13: AI-Driven Performance Analysis
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready AI-driven performance analysis including:
- Automated correlation analysis for performance metrics
- Regression-based attribution and feature importance
- Trace-based analysis (critical path, slow vs fast comparison)
- Root cause localization using topology
- Performance regression detection
- Anti-pattern detection (N+1 queries, waterfalls, etc.)
- Comparative analysis and cohort segmentation
- AI-powered performance insights with AWS Bedrock

Author: AI-Powered Observability
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetric:
    """Represents a performance metric data point"""
    timestamp: datetime
    service: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Represents a span in a distributed trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    service: str
    operation: str
    start_time: datetime
    duration_ms: float
    tags: Dict[str, Any] = field(default_factory=dict)
    error: bool = False


@dataclass
class DistributedTrace:
    """Represents a complete distributed trace"""
    trace_id: str
    spans: List[TraceSpan]
    total_duration_ms: float
    is_slow: bool  # Whether this trace is considered slow
    endpoint: str
    timestamp: datetime


@dataclass
class PerformanceInsight:
    """Represents a performance analysis insight"""
    insight_id: str
    timestamp: datetime
    insight_type: str  # correlation, regression, bottleneck, anti_pattern
    severity: str  # critical, warning, info
    service: str
    title: str
    description: str
    evidence: Dict[str, Any]
    recommended_actions: List[str]
    confidence: float


class PerformanceCorrelationAnalyzer:
    """
    Analyzes correlations between performance metrics to identify relationships.
    """
    
    def __init__(self, bedrock_client):
        """Initialize with AWS Bedrock client"""
        self.bedrock_client = bedrock_client
        self.metrics_history: Dict[str, List[PerformanceMetric]] = defaultdict(list)
    
    def add_metric(self, metric: PerformanceMetric):
        """Add metric to history"""
        key = f"{metric.service}::{metric.metric_name}"
        self.metrics_history[key].append(metric)
        
        # Keep last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        self.metrics_history[key] = [
            m for m in self.metrics_history[key]
            if m.timestamp > cutoff
        ]
    
    def analyze_correlations(self,
                           target_metric: str,
                           target_service: str,
                           min_correlation: float = 0.6) -> List[PerformanceInsight]:
        """
        Find metrics correlated with a target performance metric.
        
        Args:
            target_metric: The metric to analyze (e.g., 'latency_p95')
            target_service: Service to analyze
            min_correlation: Minimum correlation coefficient to report
            
        Returns:
            List of insights about correlated metrics
        """
        target_key = f"{target_service}::{target_metric}"
        
        if target_key not in self.metrics_history or len(self.metrics_history[target_key]) < 10:
            return []
        
        target_data = self.metrics_history[target_key]
        
        # Create timestamp-aligned data
        target_times = [m.timestamp for m in target_data]
        target_values = [m.value for m in target_data]
        
        insights = []
        correlations = []
        
        # Analyze correlations with all other metrics
        for metric_key, metric_data in self.metrics_history.items():
            if metric_key == target_key or len(metric_data) < 10:
                continue
            
            # Align data by timestamp (simplified - assumes same sampling)
            if len(metric_data) != len(target_data):
                continue
            
            other_values = [m.value for m in metric_data]
            
            # Calculate Pearson correlation
            correlation, p_value = pearsonr(target_values, other_values)
            
            if abs(correlation) >= min_correlation and p_value < 0.05:
                service, metric_name = metric_key.split('::', 1)
                correlations.append({
                    'service': service,
                    'metric': metric_name,
                    'correlation': correlation,
                    'p_value': p_value,
                    'direction': 'positive' if correlation > 0 else 'negative'
                })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        if correlations:
            # Generate insight
            top_correlations = correlations[:5]
            
            evidence = {
                'target_metric': target_metric,
                'target_service': target_service,
                'correlations': top_correlations,
                'analysis_period_hours': (target_times[-1] - target_times[0]).total_seconds() / 3600,
                'data_points': len(target_data)
            }
            
            description = self._generate_correlation_description(
                target_service,
                target_metric,
                top_correlations
            )
            
            recommended_actions = self._generate_correlation_recommendations(
                target_metric,
                top_correlations
            )
            
            insight = PerformanceInsight(
                insight_id=f"corr_{target_service}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                insight_type='correlation',
                severity='warning',
                service=target_service,
                title=f"Performance Correlation Analysis: {target_metric}",
                description=description,
                evidence=evidence,
                recommended_actions=recommended_actions,
                confidence=0.8
            )
            
            insights.append(insight)
        
        return insights
    
    def _generate_correlation_description(self,
                                         service: str,
                                         metric: str,
                                         correlations: List[Dict]) -> str:
        """Generate description using AWS Bedrock"""
        corr_text = "\n".join([
            f"- {c['service']} {c['metric']}: {c['correlation']:.2f} ({c['direction']})"
            for c in correlations[:3]
        ])
        
        prompt = f"""Analyze these performance metric correlations:

Target: {service} - {metric}

Top Correlated Metrics:
{corr_text}

Provide a brief (2-3 sentences) explanation:
1. What these correlations suggest about the performance issue
2. Which correlation is most significant
3. What should be investigated

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
    
    def _generate_correlation_recommendations(self,
                                             target_metric: str,
                                             correlations: List[Dict]) -> List[str]:
        """Generate recommended actions"""
        recommendations = []
        
        top_corr = correlations[0] if correlations else None
        
        if top_corr:
            if 'database' in top_corr['metric'].lower() or 'query' in top_corr['metric'].lower():
                recommendations.append("Investigate database query performance and connection pooling")
            
            if 'cpu' in top_corr['metric'].lower():
                recommendations.append("Analyze CPU-intensive operations and consider scaling")
            
            if 'memory' in top_corr['metric'].lower():
                recommendations.append("Check for memory leaks or inefficient memory usage")
            
            if 'cache' in top_corr['metric'].lower():
                recommendations.append("Review cache hit rates and caching strategy")
        
        recommendations.append("Profile the service to identify specific bottlenecks")
        recommendations.append("Compare performance before and after recent deployments")
        
        return recommendations


class RegressionAttributionAnalyzer:
    """
    Uses regression analysis to quantify how much each factor contributes to performance.
    """
    
    def __init__(self, bedrock_client):
        """Initialize analyzer"""
        self.bedrock_client = bedrock_client
    
    def analyze_attribution(self,
                          performance_data: List[Dict[str, float]],
                          target_metric: str = 'latency_ms') -> PerformanceInsight:
        """
        Perform regression-based attribution analysis.
        
        Args:
            performance_data: List of dicts with performance metrics
                             Each dict should have target metric + features
            target_metric: The performance metric to explain
            
        Returns:
            Insight with feature importance rankings
        """
        if len(performance_data) < 20:
            return None  # Insufficient data
        
        # Extract features and target
        feature_names = [k for k in performance_data[0].keys() if k != target_metric]
        
        X = np.array([[d[f] for f in feature_names] for d in performance_data])
        y = np.array([d[target_metric] for d in performance_data])
        
        # Train Random Forest for feature importance
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Get feature importances
        importances = rf_model.feature_importances_
        
        # Sort features by importance
        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate R-squared
        score = rf_model.score(X, y)
        
        # Generate insight
        evidence = {
            'target_metric': target_metric,
            'r_squared': score,
            'feature_importance': [
                {'feature': f, 'importance': float(imp)}
                for f, imp in feature_importance
            ],
            'samples_analyzed': len(performance_data)
        }
        
        description = self._generate_attribution_description(
            target_metric,
            feature_importance[:5],
            score
        )
        
        recommended_actions = self._generate_attribution_recommendations(
            feature_importance[:3]
        )
        
        return PerformanceInsight(
            insight_id=f"attr_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            insight_type='regression',
            severity='info',
            service='system',
            title=f"Performance Attribution Analysis: {target_metric}",
            description=description,
            evidence=evidence,
            recommended_actions=recommended_actions,
            confidence=score
        )
    
    def _generate_attribution_description(self,
                                        target: str,
                                        top_features: List[Tuple[str, float]],
                                        r_squared: float) -> str:
        """Generate attribution explanation"""
        features_text = "\n".join([
            f"- {feature}: {importance:.1%} contribution"
            for feature, importance in top_features
        ])
        
        prompt = f"""Explain this performance attribution analysis:

Target Metric: {target}
Model R²: {r_squared:.2%}

Top Contributing Factors:
{features_text}

Provide a 2-3 sentence explanation:
1. What factors most impact {target}
2. Where optimization efforts should focus
3. What the R² score tells us about model quality"""

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
    
    def _generate_attribution_recommendations(self,
                                            top_features: List[Tuple[str, float]]) -> List[str]:
        """Generate recommendations based on top contributing factors"""
        recommendations = []
        
        for feature, _ in top_features:
            if 'database' in feature.lower() or 'db' in feature.lower():
                recommendations.append(f"Optimize {feature}: Review queries, indexes, and connection pooling")
            elif 'cpu' in feature.lower():
                recommendations.append(f"Address {feature}: Profile CPU usage and optimize hot paths")
            elif 'memory' in feature.lower():
                recommendations.append(f"Reduce {feature}: Investigate memory usage patterns")
            elif 'network' in feature.lower() or 'latency' in feature.lower():
                recommendations.append(f"Improve {feature}: Check network topology and service placement")
            else:
                recommendations.append(f"Investigate {feature} impact on performance")
        
        return recommendations


class TraceAnalyzer:
    """
    Analyzes distributed traces to identify performance bottlenecks and patterns.
    """
    
    def __init__(self, bedrock_client):
        """Initialize trace analyzer"""
        self.bedrock_client = bedrock_client
        self.traces: List[DistributedTrace] = []
    
    def add_trace(self, trace: DistributedTrace):
        """Add trace to analysis pool"""
        self.traces.append(trace)
        
        # Keep last 1000 traces
        if len(self.traces) > 1000:
            self.traces = self.traces[-1000:]
    
    def analyze_critical_path(self, trace: DistributedTrace) -> Dict[str, Any]:
        """
        Identify the critical path (longest sequence) in a trace.
        
        Args:
            trace: Trace to analyze
            
        Returns:
            Critical path analysis
        """
        # Build span hierarchy
        span_children = defaultdict(list)
        spans_by_id = {span.span_id: span for span in trace.spans}
        
        for span in trace.spans:
            if span.parent_span_id:
                span_children[span.parent_span_id].append(span)
        
        # Find root spans
        root_spans = [s for s in trace.spans if not s.parent_span_id]
        
        # Compute critical path
        critical_path = []
        total_critical_time = 0
        
        def find_longest_path(span):
            nonlocal total_critical_time
            critical_path.append({
                'service': span.service,
                'operation': span.operation,
                'duration_ms': span.duration_ms
            })
            total_critical_time += span.duration_ms
            
            # Recurse to longest child
            children = span_children.get(span.span_id, [])
            if children:
                longest_child = max(children, key=lambda s: s.duration_ms)
                find_longest_path(longest_child)
        
        if root_spans:
            longest_root = max(root_spans, key=lambda s: s.duration_ms)
            find_longest_path(longest_root)
        
        return {
            'critical_path': critical_path,
            'total_time_ms': total_critical_time,
            'percentage_of_total': (total_critical_time / trace.total_duration_ms * 100) if trace.total_duration_ms > 0 else 0
        }
    
    def compare_slow_vs_fast(self,
                           endpoint: str,
                           slow_threshold_ms: float = 1000) -> Optional[PerformanceInsight]:
        """
        Compare slow traces to fast traces for the same endpoint.
        
        Args:
            endpoint: Endpoint to analyze
            slow_threshold_ms: Threshold to classify traces as slow
            
        Returns:
            Insight about differences between slow and fast traces
        """
        # Filter traces for this endpoint
        endpoint_traces = [t for t in self.traces if t.endpoint == endpoint]
        
        if len(endpoint_traces) < 10:
            return None
        
        # Classify as slow or fast
        slow_traces = [t for t in endpoint_traces if t.total_duration_ms >= slow_threshold_ms]
        fast_traces = [t for t in endpoint_traces if t.total_duration_ms < slow_threshold_ms]
        
        if not slow_traces or not fast_traces:
            return None
        
        # Analyze differences
        differences = self._identify_span_differences(slow_traces, fast_traces)
        
        if not differences:
            return None
        
        evidence = {
            'endpoint': endpoint,
            'slow_traces_count': len(slow_traces),
            'fast_traces_count': len(fast_traces),
            'slow_avg_duration_ms': np.mean([t.total_duration_ms for t in slow_traces]),
            'fast_avg_duration_ms': np.mean([t.total_duration_ms for t in fast_traces]),
            'differences': differences
        }
        
        description = self._generate_trace_comparison_description(endpoint, differences)
        
        recommended_actions = [
            f"Investigate {diff['service']} {diff['operation']}" 
            for diff in differences[:3]
        ]
        recommended_actions.append("Profile these operations to find root cause")
        
        return PerformanceInsight(
            insight_id=f"trace_{endpoint}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            insight_type='bottleneck',
            severity='warning',
            service=endpoint,
            title=f"Slow Trace Analysis: {endpoint}",
            description=description,
            evidence=evidence,
            recommended_actions=recommended_actions,
            confidence=0.85
        )
    
    def _identify_span_differences(self,
                                  slow_traces: List[DistributedTrace],
                                  fast_traces: List[DistributedTrace]) -> List[Dict]:
        """Identify operations that are slower in slow traces"""
        # Collect span durations by service and operation
        slow_durations = defaultdict(list)
        fast_durations = defaultdict(list)
        
        for trace in slow_traces:
            for span in trace.spans:
                key = f"{span.service}::{span.operation}"
                slow_durations[key].append(span.duration_ms)
        
        for trace in fast_traces:
            for span in trace.spans:
                key = f"{span.service}::{span.operation}"
                fast_durations[key].append(span.duration_ms)
        
        # Compare averages
        differences = []
        
        for key in slow_durations:
            if key in fast_durations:
                slow_avg = np.mean(slow_durations[key])
                fast_avg = np.mean(fast_durations[key])
                
                # Calculate difference
                if fast_avg > 0:
                    difference_pct = ((slow_avg - fast_avg) / fast_avg) * 100
                    
                    if difference_pct > 50:  # At least 50% slower
                        service, operation = key.split('::', 1)
                        differences.append({
                            'service': service,
                            'operation': operation,
                            'slow_avg_ms': slow_avg,
                            'fast_avg_ms': fast_avg,
                            'difference_pct': difference_pct
                        })
        
        # Sort by difference percentage
        differences.sort(key=lambda x: x['difference_pct'], reverse=True)
        
        return differences[:5]
    
    def _generate_trace_comparison_description(self,
                                              endpoint: str,
                                              differences: List[Dict]) -> str:
        """Generate trace comparison description"""
        diffs_text = "\n".join([
            f"- {d['service']} {d['operation']}: {d['difference_pct']:.0f}% slower ({d['slow_avg_ms']:.0f}ms vs {d['fast_avg_ms']:.0f}ms)"
            for d in differences[:3]
        ])
        
        prompt = f"""Analyze these performance differences between slow and fast traces:

Endpoint: {endpoint}

Operations That Are Slower:
{diffs_text}

Provide a 2-3 sentence analysis:
1. What operations contribute most to slow traces
2. What this suggests about the bottleneck
3. Where to focus investigation"""

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
    
    def detect_anti_patterns(self, trace: DistributedTrace) -> List[PerformanceInsight]:
        """
        Detect common anti-patterns in traces (N+1 queries, waterfalls, etc.)
        
        Args:
            trace: Trace to analyze
            
        Returns:
            List of detected anti-pattern insights
        """
        insights = []
        
        # Detect N+1 query pattern
        n_plus_1 = self._detect_n_plus_1(trace)
        if n_plus_1:
            insights.append(n_plus_1)
        
        # Detect synchronous waterfall
        waterfall = self._detect_waterfall(trace)
        if waterfall:
            insights.append(waterfall)
        
        return insights
    
    def _detect_n_plus_1(self, trace: DistributedTrace) -> Optional[PerformanceInsight]:
        """Detect N+1 query anti-pattern"""
        # Look for many sequential calls to the same service/operation
        span_sequence = []
        for span in sorted(trace.spans, key=lambda s: s.start_time):
            span_sequence.append(f"{span.service}::{span.operation}")
        
        # Count consecutive calls to same operation
        max_consecutive = 0
        current_op = None
        current_count = 0
        problem_op = None
        
        for op in span_sequence:
            if op == current_op:
                current_count += 1
                if current_count > max_consecutive:
                    max_consecutive = current_count
                    problem_op = op
            else:
                current_op = op
                current_count = 1
        
        # N+1 if we see many consecutive calls (threshold: 5+)
        if max_consecutive >= 5:
            service, operation = problem_op.split('::', 1)
            
            return PerformanceInsight(
                insight_id=f"n_plus_1_{trace.trace_id}",
                timestamp=datetime.now(),
                insight_type='anti_pattern',
                severity='warning',
                service=service,
                title="N+1 Query Pattern Detected",
                description=f"Detected {max_consecutive} consecutive calls to {service} {operation}. This N+1 pattern suggests queries that could be batched.",
                evidence={
                    'consecutive_calls': max_consecutive,
                    'service': service,
                    'operation': operation,
                    'trace_id': trace.trace_id
                },
                recommended_actions=[
                    f"Batch calls to {service} {operation} to reduce round trips",
                    "Review code for loops making individual queries",
                    "Consider using batch APIs or data loading strategies"
                ],
                confidence=0.9
            )
        
        return None
    
    def _detect_waterfall(self, trace: DistributedTrace) -> Optional[PerformanceInsight]:
        """Detect synchronous waterfall anti-pattern"""
        # Look for sequential spans that could be parallelized
        # Build parent-child relationships
        span_children = defaultdict(list)
        
        for span in trace.spans:
            if span.parent_span_id:
                span_children[span.parent_span_id].append(span)
        
        # Find spans with multiple sequential children
        for parent_id, children in span_children.items():
            if len(children) >= 3:
                # Check if children are sequential (non-overlapping)
                sorted_children = sorted(children, key=lambda s: s.start_time)
                
                is_sequential = True
                for i in range(len(sorted_children) - 1):
                    child1_end = sorted_children[i].start_time + timedelta(milliseconds=sorted_children[i].duration_ms)
                    child2_start = sorted_children[i + 1].start_time
                    
                    # Allow small overlap (50ms) for timing precision
                    if child2_start < child1_end - timedelta(milliseconds=50):
                        is_sequential = False
                        break
                
                if is_sequential:
                    total_sequential_time = sum(c.duration_ms for c in sorted_children)
                    
                    return PerformanceInsight(
                        insight_id=f"waterfall_{trace.trace_id}",
                        timestamp=datetime.now(),
                        insight_type='anti_pattern',
                        severity='warning',
                        service=trace.endpoint,
                        title="Synchronous Waterfall Detected",
                        description=f"Detected {len(sorted_children)} sequential service calls totaling {total_sequential_time:.0f}ms that could potentially be parallelized.",
                        evidence={
                            'sequential_calls': len(sorted_children),
                            'total_time_ms': total_sequential_time,
                            'services': [c.service for c in sorted_children],
                            'trace_id': trace.trace_id
                        },
                        recommended_actions=[
                            "Consider parallelizing independent service calls",
                            "Use async/await or Promise.all patterns",
                            "Review if all calls are necessary or can be optimized"
                        ],
                        confidence=0.8
                    )
        
        return None


class PerformanceRegressionDetector:
    """
    Detects performance regressions by comparing before/after deployments.
    """
    
    def __init__(self, bedrock_client):
        """Initialize regression detector"""
        self.bedrock_client = bedrock_client
    
    def detect_regression(self,
                        before_metrics: List[float],
                        after_metrics: List[float],
                        metric_name: str,
                        service: str,
                        deployment_id: str) -> Optional[PerformanceInsight]:
        """
        Detect performance regression using statistical comparison.
        
        Args:
            before_metrics: Metric values before deployment
            after_metrics: Metric values after deployment
            metric_name: Name of the metric
            service: Service name
            deployment_id: Deployment identifier
            
        Returns:
            Insight if regression detected
        """
        if len(before_metrics) < 10 or len(after_metrics) < 10:
            return None  # Insufficient data
        
        # Calculate statistics
        before_mean = np.mean(before_metrics)
        after_mean = np.mean(after_metrics)
        
        before_p95 = np.percentile(before_metrics, 95)
        after_p95 = np.percentile(after_metrics, 95)
        
        # Percentage change
        mean_change_pct = ((after_mean - before_mean) / before_mean) * 100
        p95_change_pct = ((after_p95 - before_p95) / before_p95) * 100
        
        # Statistical significance test (t-test)
        t_stat, p_value = stats.ttest_ind(before_metrics, after_metrics)
        
        # Detect regression (significant increase in latency-like metrics)
        is_regression = (
            mean_change_pct > 10 and  # At least 10% increase
            p_value < 0.05  # Statistically significant
        )
        
        if is_regression:
            severity = 'critical' if mean_change_pct > 50 else 'warning'
            
            evidence = {
                'metric_name': metric_name,
                'before_mean': before_mean,
                'after_mean': after_mean,
                'mean_change_pct': mean_change_pct,
                'before_p95': before_p95,
                'after_p95': after_p95,
                'p95_change_pct': p95_change_pct,
                'p_value': p_value,
                'deployment_id': deployment_id,
                'before_samples': len(before_metrics),
                'after_samples': len(after_metrics)
            }
            
            description = self._generate_regression_description(
                service,
                metric_name,
                mean_change_pct,
                deployment_id
            )
            
            return PerformanceInsight(
                insight_id=f"regression_{deployment_id}",
                timestamp=datetime.now(),
                insight_type='regression',
                severity=severity,
                service=service,
                title=f"Performance Regression Detected: {metric_name}",
                description=description,
                evidence=evidence,
                recommended_actions=[
                    f"Review changes in deployment {deployment_id}",
                    "Compare traces before and after deployment",
                    "Profile the service to identify new bottlenecks",
                    "Consider rollback if regression is severe"
                ],
                confidence=1 - p_value  # Higher confidence with lower p-value
            )
        
        return None
    
    def _generate_regression_description(self,
                                       service: str,
                                       metric: str,
                                       change_pct: float,
                                       deployment_id: str) -> str:
        """Generate regression description"""
        prompt = f"""Explain this performance regression:

Service: {service}
Metric: {metric}
Change: {change_pct:+.1f}%
Deployment: {deployment_id}

Provide 2-3 sentences explaining:
1. The severity of this regression
2. Likely causes to investigate
3. Immediate actions needed"""

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


class PerformanceAnalysisPipeline:
    """
    Production pipeline orchestrating all performance analysis approaches.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize performance analysis pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.correlation_analyzer = PerformanceCorrelationAnalyzer(self.bedrock_client)
        self.attribution_analyzer = RegressionAttributionAnalyzer(self.bedrock_client)
        self.trace_analyzer = TraceAnalyzer(self.bedrock_client)
        self.regression_detector = PerformanceRegressionDetector(self.bedrock_client)
        
        self.insights: List[PerformanceInsight] = []
    
    def process_metrics(self, metrics: List[PerformanceMetric]):
        """Process performance metrics"""
        for metric in metrics:
            self.correlation_analyzer.add_metric(metric)
    
    def process_trace(self, trace: DistributedTrace):
        """Process distributed trace"""
        self.trace_analyzer.add_trace(trace)
    
    def analyze_performance_issue(self,
                                 service: str,
                                 metric: str) -> List[PerformanceInsight]:
        """
        Comprehensive analysis of a performance issue.
        
        Args:
            service: Service experiencing performance issues
            metric: Performance metric of concern
            
        Returns:
            List of insights
        """
        insights = []
        
        # Correlation analysis
        corr_insights = self.correlation_analyzer.analyze_correlations(
            metric,
            service
        )
        insights.extend(corr_insights)
        
        # Store insights
        self.insights.extend(insights)
        
        return insights
    
    def analyze_traces_for_endpoint(self,
                                   endpoint: str) -> List[PerformanceInsight]:
        """
        Analyze traces for a specific endpoint.
        
        Args:
            endpoint: Endpoint to analyze
            
        Returns:
            List of insights
        """
        insights = []
        
        # Slow vs fast comparison
        comparison_insight = self.trace_analyzer.compare_slow_vs_fast(endpoint)
        if comparison_insight:
            insights.append(comparison_insight)
        
        # Check recent traces for anti-patterns
        recent_traces = [t for t in self.trace_analyzer.traces if t.endpoint == endpoint][-10:]
        
        for trace in recent_traces:
            anti_pattern_insights = self.trace_analyzer.detect_anti_patterns(trace)
            insights.extend(anti_pattern_insights)
        
        self.insights.extend(insights)
        
        return insights
    
    def get_all_insights(self, min_severity: str = 'info') -> List[PerformanceInsight]:
        """Get all generated insights"""
        severity_order = {'info': 0, 'warning': 1, 'critical': 2}
        min_level = severity_order.get(min_severity, 0)
        
        return [
            insight for insight in self.insights
            if severity_order.get(insight.severity, 0) >= min_level
        ]


# Example usage and realistic data generators
def generate_sample_performance_data() -> List[Dict[str, float]]:
    """Generate sample performance data for attribution analysis"""
    data = []
    
    for i in range(100):
        # Simulate relationships between factors and latency
        cpu_usage = np.random.uniform(20, 90)
        memory_usage = np.random.uniform(40, 85)
        db_query_time = np.random.uniform(10, 200)
        cache_hit_rate = np.random.uniform(60, 98)
        request_rate = np.random.uniform(100, 500)
        
        # Latency is influenced by these factors
        latency = (
            50 +  # Base latency
            cpu_usage * 0.5 +  # CPU impact
            memory_usage * 0.3 +  # Memory impact
            db_query_time * 2 +  # Database is big factor
            (100 - cache_hit_rate) * 3 +  # Cache misses hurt
            request_rate * 0.1 +  # Load impact
            np.random.normal(0, 20)  # Noise
        )
        
        data.append({
            'latency_ms': max(0, latency),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'db_query_time_ms': db_query_time,
            'cache_hit_rate': cache_hit_rate,
            'request_rate': request_rate
        })
    
    return data


def generate_sample_trace(trace_id: str,
                         is_slow: bool = False,
                         has_n_plus_1: bool = False) -> DistributedTrace:
    """Generate sample distributed trace"""
    spans = []
    start_time = datetime.now()
    
    # Root span
    root_duration = 800 if is_slow else 200
    spans.append(TraceSpan(
        trace_id=trace_id,
        span_id='span_001',
        parent_span_id=None,
        service='api-gateway',
        operation='POST /checkout',
        start_time=start_time,
        duration_ms=root_duration
    ))
    
    # Auth service call
    auth_start = start_time + timedelta(milliseconds=10)
    auth_duration = 50
    spans.append(TraceSpan(
        trace_id=trace_id,
        span_id='span_002',
        parent_span_id='span_001',
        service='auth-service',
        operation='verify_token',
        start_time=auth_start,
        duration_ms=auth_duration
    ))
    
    if has_n_plus_1:
        # Simulate N+1 pattern - many sequential database calls
        current_time = auth_start + timedelta(milliseconds=auth_duration + 10)
        for i in range(10):
            spans.append(TraceSpan(
                trace_id=trace_id,
                span_id=f'span_db_{i:03d}',
                parent_span_id='span_001',
                service='database',
                operation='SELECT product',
                start_time=current_time,
                duration_ms=30
            ))
            current_time += timedelta(milliseconds=35)
    else:
        # Normal database call
        db_start = auth_start + timedelta(milliseconds=auth_duration + 10)
        db_duration = 150 if is_slow else 50
        spans.append(TraceSpan(
            trace_id=trace_id,
            span_id='span_003',
            parent_span_id='span_001',
            service='database',
            operation='SELECT order_details',
            start_time=db_start,
            duration_ms=db_duration
        ))
    
    return DistributedTrace(
        trace_id=trace_id,
        spans=spans,
        total_duration_ms=root_duration,
        is_slow=is_slow,
        endpoint='POST /checkout',
        timestamp=start_time
    )


def main():
    """
    Demonstrate AI-driven performance analysis pipeline.
    """
    print("=" * 80)
    print("Chapter 13: AI-Driven Performance Analysis")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing performance analysis pipeline...")
    pipeline = PerformanceAnalysisPipeline(aws_region='us-east-1')
    print()
    
    # Scenario 1: Correlation Analysis
    print("=" * 80)
    print("SCENARIO 1: Performance Correlation Analysis")
    print("=" * 80)
    print()
    
    # Generate sample metrics
    base_time = datetime.now() - timedelta(hours=24)
    for i in range(100):
        timestamp = base_time + timedelta(minutes=15*i)
        
        # Latency correlated with database query time
        db_query_time = 50 + i * 0.5 + np.random.normal(0, 5)
        latency = 100 + db_query_time * 1.5 + np.random.normal(0, 10)
        
        pipeline.process_metrics([
            PerformanceMetric(timestamp, 'checkout-service', 'latency_p95', latency),
            PerformanceMetric(timestamp, 'checkout-service', 'db_query_time_ms', db_query_time),
            PerformanceMetric(timestamp, 'checkout-service', 'cpu_usage', np.random.uniform(40, 70))
        ])
    
    print("Analyzing performance correlations...")
    corr_insights = pipeline.analyze_performance_issue('checkout-service', 'latency_p95')
    
    for insight in corr_insights:
        print(f"\n📊 INSIGHT: {insight.title}")
        print(f"   Severity: {insight.severity.upper()}")
        print(f"   Confidence: {insight.confidence:.0%}")
        print(f"\n   {insight.description}")
        print(f"\n   Recommended Actions:")
        for action in insight.recommended_actions:
            print(f"     • {action}")
    
    # Scenario 2: Attribution Analysis
    print("\n" + "=" * 80)
    print("SCENARIO 2: Regression-Based Attribution")
    print("=" * 80)
    print()
    
    print("Analyzing performance factor contributions...")
    perf_data = generate_sample_performance_data()
    attr_insight = pipeline.attribution_analyzer.analyze_attribution(perf_data)
    
    if attr_insight:
        print(f"\n📊 ATTRIBUTION ANALYSIS")
        print(f"   Target: {attr_insight.evidence['target_metric']}")
        print(f"   Model R²: {attr_insight.evidence['r_squared']:.2%}")
        print(f"\n   Top Contributing Factors:")
        for feat in attr_insight.evidence['feature_importance'][:5]:
            print(f"     • {feat['feature']}: {feat['importance']:.1%}")
        print(f"\n   {attr_insight.description}")
    
    # Scenario 3: Trace Analysis
    print("\n" + "=" * 80)
    print("SCENARIO 3: Trace-Based Performance Analysis")
    print("=" * 80)
    print()
    
    # Generate sample traces
    print("Processing distributed traces...")
    for i in range(20):
        is_slow = i % 5 == 0  # Every 5th trace is slow
        trace = generate_sample_trace(f"trace_{i:03d}", is_slow=is_slow)
        pipeline.process_trace(trace)
    
    # Add trace with N+1 pattern
    n_plus_1_trace = generate_sample_trace("trace_n_plus_1", has_n_plus_1=True)
    pipeline.process_trace(n_plus_1_trace)
    
    trace_insights = pipeline.analyze_traces_for_endpoint('POST /checkout')
    
    for insight in trace_insights:
        print(f"\n🔍 TRACE INSIGHT: {insight.title}")
        print(f"   Type: {insight.insight_type}")
        print(f"   Severity: {insight.severity.upper()}")
        print(f"\n   {insight.description}")
        if 'differences' in insight.evidence:
            print(f"\n   Performance Differences:")
            for diff in insight.evidence['differences'][:3]:
                print(f"     • {diff['service']} {diff['operation']}: {diff['difference_pct']:.0f}% slower")
    
    # Scenario 4: Regression Detection
    print("\n" + "=" * 80)
    print("SCENARIO 4: Performance Regression Detection")
    print("=" * 80)
    print()
    
    # Simulate before/after deployment metrics
    before = [np.random.normal(200, 20) for _ in range(50)]
    after = [np.random.normal(280, 25) for _ in range(50)]  # 40% regression
    
    print("Comparing performance before and after deployment...")
    regression = pipeline.regression_detector.detect_regression(
        before,
        after,
        'latency_p95',
        'payment-service',
        'deploy_v2.4.1'
    )
    
    if regression:
        print(f"\n⚠️  REGRESSION DETECTED")
        print(f"   Service: {regression.service}")
        print(f"   Metric: {regression.evidence['metric_name']}")
        print(f"   Before: {regression.evidence['before_mean']:.1f}ms")
        print(f"   After: {regression.evidence['after_mean']:.1f}ms")
        print(f"   Change: {regression.evidence['mean_change_pct']:+.1f}%")
        print(f"   P-value: {regression.evidence['p_value']:.4f}")
        print(f"\n   {regression.description}")
    
    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE INSIGHTS SUMMARY")
    print("=" * 80)
    print()
    
    all_insights = pipeline.get_all_insights()
    
    by_type = Counter(i.insight_type for i in all_insights)
    by_severity = Counter(i.severity for i in all_insights)
    
    print(f"Total Insights: {len(all_insights)}")
    print(f"\nBy Type:")
    for insight_type, count in by_type.items():
        print(f"  {insight_type}: {count}")
    
    print(f"\nBy Severity:")
    for severity, count in by_severity.items():
        print(f"  {severity}: {count}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
