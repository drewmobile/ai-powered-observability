"""
Chapter 14: Capacity Planning with ML
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready ML-based capacity planning including:
- Time series demand forecasting with seasonality handling
- Capacity modeling and utilization analysis
- Multi-resource capacity tracking (CPU, memory, storage, network)
- Bottleneck identification and prediction
- Scaling recommendations with cost optimization
- Predictive auto-scaling suggestions
- AI-powered capacity insights with AWS Bedrock

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
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DemandDataPoint:
    """Represents a demand measurement"""
    timestamp: datetime
    service: str
    requests_per_second: float
    active_users: int
    transaction_volume: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUtilization:
    """Represents resource utilization at a point in time"""
    timestamp: datetime
    service: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_mbps: float
    active_connections: int
    instance_count: int


@dataclass
class CapacityLimit:
    """Represents capacity limits for a resource"""
    service: str
    resource_type: str  # cpu, memory, disk, network, connections
    current_capacity: float
    max_capacity: float
    unit: str
    cost_per_unit: float


@dataclass
class ScalingRecommendation:
    """Represents a scaling recommendation"""
    recommendation_id: str
    timestamp: datetime
    service: str
    action: str  # scale_up, scale_down, right_size
    resource_type: str
    current_value: float
    recommended_value: float
    urgency: str  # critical, high, medium, low
    lead_time_hours: float
    cost_impact_monthly: float
    rationale: str
    confidence: float


@dataclass
class CapacityInsight:
    """Represents a capacity planning insight"""
    insight_id: str
    timestamp: datetime
    insight_type: str  # forecast, bottleneck, optimization, headroom
    service: str
    title: str
    description: str
    evidence: Dict[str, Any]
    recommendations: List[str]
    priority: str  # critical, high, medium, low


class DemandForecaster:
    """
    Forecasts future demand using time series analysis with seasonality.
    """
    
    def __init__(self, bedrock_client):
        """Initialize demand forecaster"""
        self.bedrock_client = bedrock_client
        self.demand_history: Dict[str, List[DemandDataPoint]] = defaultdict(list)
    
    def add_demand_data(self, data_point: DemandDataPoint):
        """Add demand data point to history"""
        key = data_point.service
        self.demand_history[key].append(data_point)
        
        # Keep last 90 days
        cutoff = datetime.now() - timedelta(days=90)
        self.demand_history[key] = [
            dp for dp in self.demand_history[key]
            if dp.timestamp > cutoff
        ]
    
    def forecast_demand(self,
                       service: str,
                       forecast_hours: int = 168) -> Optional[Dict[str, Any]]:
        """
        Forecast future demand for a service.
        
        Args:
            service: Service to forecast
            forecast_hours: Hours to forecast ahead (default 1 week)
            
        Returns:
            Forecast with confidence intervals
        """
        if service not in self.demand_history or len(self.demand_history[service]) < 48:
            return None  # Insufficient data (need at least 2 days)
        
        data = self.demand_history[service]
        
        # Extract time series
        timestamps = [dp.timestamp for dp in data]
        demand_values = [dp.requests_per_second for dp in data]
        
        # Fit trend with seasonality
        # Convert to hours since start
        hours_since_start = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
        X = np.array(hours_since_start).reshape(-1, 1)
        y = np.array(demand_values)
        
        # Add hourly seasonality features (sin/cos for 24-hour cycle)
        X_seasonal = np.column_stack([
            X,
            np.sin(2 * np.pi * X / 24),  # Daily pattern
            np.cos(2 * np.pi * X / 24),
            np.sin(2 * np.pi * X / 168),  # Weekly pattern
            np.cos(2 * np.pi * X / 168)
        ])
        
        # Fit model
        model = LinearRegression()
        model.fit(X_seasonal, y)
        
        # Generate forecasts
        last_hour = hours_since_start[-1]
        forecast_hours_array = np.arange(last_hour + 1, last_hour + 1 + forecast_hours)
        
        X_forecast = np.column_stack([
            forecast_hours_array.reshape(-1, 1),
            np.sin(2 * np.pi * forecast_hours_array / 24),
            np.cos(2 * np.pi * forecast_hours_array / 24),
            np.sin(2 * np.pi * forecast_hours_array / 168),
            np.cos(2 * np.pi * forecast_hours_array / 168)
        ])
        
        predictions = model.predict(X_forecast)
        
        # Calculate prediction intervals (simplified)
        residuals = y - model.predict(X_seasonal)
        std_error = np.std(residuals)
        
        # 95% confidence interval
        lower_bound = predictions - 1.96 * std_error
        upper_bound = predictions + 1.96 * std_error
        
        # Calculate R-squared for confidence
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Find peak demand in forecast
        peak_demand = np.max(predictions)
        peak_hour = forecast_hours_array[np.argmax(predictions)]
        peak_time = timestamps[0] + timedelta(hours=float(peak_hour))
        
        return {
            'service': service,
            'forecast_horizon_hours': forecast_hours,
            'predictions': predictions.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'peak_demand': float(peak_demand),
            'peak_time': peak_time,
            'current_demand': demand_values[-1],
            'growth_rate_percent': ((predictions[-1] - demand_values[-1]) / demand_values[-1] * 100) if demand_values[-1] > 0 else 0,
            'confidence': r_squared,
            'forecast_timestamps': [timestamps[0] + timedelta(hours=h) for h in forecast_hours_array]
        }
    
    def detect_growth_trend(self, service: str) -> Optional[Dict[str, Any]]:
        """
        Detect and quantify growth trends.
        
        Args:
            service: Service to analyze
            
        Returns:
            Growth trend analysis
        """
        if service not in self.demand_history or len(self.demand_history[service]) < 72:
            return None  # Need at least 3 days
        
        data = self.demand_history[service][-168:]  # Last week
        
        # Extract trend
        hours = [(dp.timestamp - data[0].timestamp).total_seconds() / 3600 for dp in data]
        demand = [dp.requests_per_second for dp in data]
        
        X = np.array(hours).reshape(-1, 1)
        y = np.array(demand)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate growth rate
        growth_per_hour = model.coef_[0]
        growth_per_day = growth_per_hour * 24
        growth_percent_per_week = (growth_per_day * 7 / y[0] * 100) if y[0] > 0 else 0
        
        # Calculate R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        is_significant_growth = abs(growth_percent_per_week) > 5 and r_squared > 0.5
        
        return {
            'service': service,
            'growth_per_hour': growth_per_hour,
            'growth_per_day': growth_per_day,
            'growth_percent_per_week': growth_percent_per_week,
            'is_significant': is_significant_growth,
            'confidence': r_squared,
            'current_demand': demand[-1],
            'projected_demand_next_week': demand[-1] + growth_per_day * 7
        }


class CapacityModeler:
    """
    Models capacity requirements and identifies bottlenecks.
    """
    
    def __init__(self, bedrock_client):
        """Initialize capacity modeler"""
        self.bedrock_client = bedrock_client
        self.utilization_history: Dict[str, List[ResourceUtilization]] = defaultdict(list)
        self.capacity_limits: Dict[str, Dict[str, CapacityLimit]] = defaultdict(dict)
    
    def add_utilization_data(self, utilization: ResourceUtilization):
        """Add resource utilization data"""
        key = utilization.service
        self.utilization_history[key].append(utilization)
        
        # Keep last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        self.utilization_history[key] = [
            u for u in self.utilization_history[key]
            if u.timestamp > cutoff
        ]
    
    def set_capacity_limit(self, limit: CapacityLimit):
        """Set capacity limit for a resource"""
        self.capacity_limits[limit.service][limit.resource_type] = limit
    
    def analyze_capacity_headroom(self, service: str) -> Optional[CapacityInsight]:
        """
        Analyze available capacity headroom for all resources.
        
        Args:
            service: Service to analyze
            
        Returns:
            Capacity headroom insight
        """
        if service not in self.utilization_history or service not in self.capacity_limits:
            return None
        
        recent_util = self.utilization_history[service][-10:]  # Last 10 data points
        
        if not recent_util:
            return None
        
        # Calculate average utilization for each resource
        avg_cpu = np.mean([u.cpu_percent for u in recent_util])
        avg_memory = np.mean([u.memory_percent for u in recent_util])
        avg_disk = np.mean([u.disk_percent for u in recent_util])
        avg_network = np.mean([u.network_mbps for u in recent_util])
        
        # Identify resources with low headroom (>80% utilization)
        low_headroom = []
        
        if avg_cpu > 80:
            low_headroom.append(f"CPU at {avg_cpu:.1f}%")
        if avg_memory > 80:
            low_headroom.append(f"Memory at {avg_memory:.1f}%")
        if avg_disk > 80:
            low_headroom.append(f"Disk at {avg_disk:.1f}%")
        
        # Determine priority
        if low_headroom:
            max_util = max(avg_cpu, avg_memory, avg_disk)
            priority = 'critical' if max_util > 90 else 'high'
        else:
            priority = 'medium'
        
        evidence = {
            'cpu_utilization': avg_cpu,
            'memory_utilization': avg_memory,
            'disk_utilization': avg_disk,
            'network_utilization_mbps': avg_network,
            'instance_count': recent_util[-1].instance_count,
            'low_headroom_resources': low_headroom
        }
        
        description = self._generate_headroom_description(service, evidence)
        
        recommendations = []
        if avg_cpu > 80:
            recommendations.append("Scale CPU capacity or optimize CPU-intensive operations")
        if avg_memory > 80:
            recommendations.append("Increase memory limits or investigate memory usage")
        if avg_disk > 80:
            recommendations.append("Expand disk capacity or implement cleanup procedures")
        if not low_headroom:
            recommendations.append("Current capacity headroom is adequate")
        
        return CapacityInsight(
            insight_id=f"headroom_{service}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            insight_type='headroom',
            service=service,
            title=f"Capacity Headroom Analysis: {service}",
            description=description,
            evidence=evidence,
            recommendations=recommendations,
            priority=priority
        )
    
    def identify_bottleneck(self, service: str) -> Optional[CapacityInsight]:
        """
        Identify which resource is the bottleneck.
        
        Args:
            service: Service to analyze
            
        Returns:
            Bottleneck identification insight
        """
        if service not in self.utilization_history or service not in self.capacity_limits:
            return None
        
        recent_util = self.utilization_history[service][-10:]
        
        if not recent_util:
            return None
        
        # Calculate headroom for each resource (100% - utilization)
        cpu_headroom = 100 - np.mean([u.cpu_percent for u in recent_util])
        memory_headroom = 100 - np.mean([u.memory_percent for u in recent_util])
        disk_headroom = 100 - np.mean([u.disk_percent for u in recent_util])
        
        # The resource with least headroom is the bottleneck
        headrooms = {
            'CPU': cpu_headroom,
            'Memory': memory_headroom,
            'Disk': disk_headroom
        }
        
        bottleneck = min(headrooms, key=headrooms.get)
        bottleneck_headroom = headrooms[bottleneck]
        
        evidence = {
            'bottleneck_resource': bottleneck,
            'bottleneck_headroom_percent': bottleneck_headroom,
            'resource_headrooms': headrooms
        }
        
        description = self._generate_bottleneck_description(service, bottleneck, bottleneck_headroom)
        
        recommendations = [
            f"Prioritize scaling {bottleneck} capacity",
            f"Investigate {bottleneck} usage patterns for optimization opportunities",
            "Monitor bottleneck resource closely during peak periods"
        ]
        
        priority = 'critical' if bottleneck_headroom < 10 else 'high' if bottleneck_headroom < 20 else 'medium'
        
        return CapacityInsight(
            insight_id=f"bottleneck_{service}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            insight_type='bottleneck',
            service=service,
            title=f"Bottleneck Identified: {bottleneck}",
            description=description,
            evidence=evidence,
            recommendations=recommendations,
            priority=priority
        )
    
    def _generate_headroom_description(self, service: str, evidence: Dict) -> str:
        """Generate headroom description using AWS Bedrock"""
        prompt = f"""Analyze capacity headroom for this service:

Service: {service}
CPU Utilization: {evidence['cpu_utilization']:.1f}%
Memory Utilization: {evidence['memory_utilization']:.1f}%
Disk Utilization: {evidence['disk_utilization']:.1f}%
Instance Count: {evidence['instance_count']}

Low Headroom Resources: {', '.join(evidence['low_headroom_resources']) if evidence['low_headroom_resources'] else 'None'}

Provide a 2-3 sentence analysis:
1. Overall capacity health
2. Most concerning resource if any
3. Immediate action needed if any"""

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
    
    def _generate_bottleneck_description(self, service: str, bottleneck: str, headroom: float) -> str:
        """Generate bottleneck description"""
        prompt = f"""Explain this capacity bottleneck:

Service: {service}
Bottleneck Resource: {bottleneck}
Available Headroom: {headroom:.1f}%

Provide 2-3 sentences:
1. Why this resource is the bottleneck
2. Impact of the bottleneck
3. What should be prioritized"""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 200,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']


class ScalingOptimizer:
    """
    Generates scaling recommendations based on demand and capacity analysis.
    """
    
    def __init__(self, bedrock_client):
        """Initialize scaling optimizer"""
        self.bedrock_client = bedrock_client
    
    def generate_scaling_recommendation(self,
                                       service: str,
                                       demand_forecast: Dict[str, Any],
                                       current_capacity: float,
                                       target_utilization: float = 70.0,
                                       cost_per_unit: float = 100.0) -> ScalingRecommendation:
        """
        Generate scaling recommendation based on forecast.
        
        Args:
            service: Service name
            demand_forecast: Demand forecast from DemandForecaster
            current_capacity: Current capacity (e.g., requests/sec)
            target_utilization: Target utilization percentage
            cost_per_unit: Cost per capacity unit per month
            
        Returns:
            Scaling recommendation
        """
        peak_demand = demand_forecast['peak_demand']
        current_demand = demand_forecast['current_demand']
        
        # Calculate required capacity for peak demand
        required_capacity = peak_demand / (target_utilization / 100)
        
        # Determine action
        capacity_gap = required_capacity - current_capacity
        capacity_gap_percent = (capacity_gap / current_capacity * 100) if current_capacity > 0 else 0
        
        if capacity_gap > 0:
            action = 'scale_up'
            urgency = self._determine_urgency(demand_forecast, capacity_gap_percent)
        elif capacity_gap < -0.2 * current_capacity:  # More than 20% over-capacity
            action = 'scale_down'
            urgency = 'low'
        else:
            action = 'maintain'
            urgency = 'low'
        
        # Calculate cost impact
        capacity_change = required_capacity - current_capacity
        cost_impact = capacity_change * cost_per_unit
        
        # Calculate lead time (time until peak)
        peak_time = demand_forecast['peak_time']
        lead_time_hours = (peak_time - datetime.now()).total_seconds() / 3600
        
        rationale = self._generate_scaling_rationale(
            service,
            action,
            current_capacity,
            required_capacity,
            peak_demand,
            lead_time_hours
        )
        
        return ScalingRecommendation(
            recommendation_id=f"scale_{service}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            service=service,
            action=action,
            resource_type='instances',
            current_value=current_capacity,
            recommended_value=required_capacity,
            urgency=urgency,
            lead_time_hours=lead_time_hours,
            cost_impact_monthly=cost_impact,
            rationale=rationale,
            confidence=demand_forecast['confidence']
        )
    
    def _determine_urgency(self, forecast: Dict, gap_percent: float) -> str:
        """Determine urgency of scaling action"""
        peak_time = forecast['peak_time']
        hours_to_peak = (peak_time - datetime.now()).total_seconds() / 3600
        
        if gap_percent > 50 and hours_to_peak < 4:
            return 'critical'
        elif gap_percent > 30 and hours_to_peak < 12:
            return 'high'
        elif gap_percent > 20 or hours_to_peak < 24:
            return 'medium'
        else:
            return 'low'
    
    def _generate_scaling_rationale(self,
                                   service: str,
                                   action: str,
                                   current: float,
                                   recommended: float,
                                   peak_demand: float,
                                   lead_time: float) -> str:
        """Generate scaling rationale"""
        prompt = f"""Explain this capacity scaling recommendation:

Service: {service}
Action: {action}
Current Capacity: {current:.0f} req/sec
Recommended Capacity: {recommended:.0f} req/sec
Expected Peak Demand: {peak_demand:.0f} req/sec
Time to Peak: {lead_time:.1f} hours

Provide 2-3 sentences explaining:
1. Why this scaling is needed
2. Timing and urgency
3. Expected benefit"""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 200,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
    
    def identify_right_sizing_opportunities(self,
                                          utilization_data: List[ResourceUtilization],
                                          service: str) -> List[ScalingRecommendation]:
        """
        Identify right-sizing opportunities (over or under-provisioned resources).
        
        Args:
            utilization_data: Historical utilization data
            service: Service to analyze
            
        Returns:
            List of right-sizing recommendations
        """
        if len(utilization_data) < 24:  # Need at least a day of data
            return []
        
        recommendations = []
        
        # Analyze average utilization
        avg_cpu = np.mean([u.cpu_percent for u in utilization_data])
        avg_memory = np.mean([u.memory_percent for u in utilization_data])
        
        # Check for over-provisioning (consistently low utilization)
        if avg_cpu < 30 and avg_memory < 40:
            recommendations.append(ScalingRecommendation(
                recommendation_id=f"rightsize_{service}_down_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                service=service,
                action='right_size',
                resource_type='instance_type',
                current_value=utilization_data[-1].instance_count,
                recommended_value=utilization_data[-1].instance_count * 0.7,
                urgency='medium',
                lead_time_hours=24,
                cost_impact_monthly=-3000,  # Estimated savings
                rationale=f"CPU averaging {avg_cpu:.1f}% and memory {avg_memory:.1f}% indicate over-provisioning. Consider smaller instance types.",
                confidence=0.8
            ))
        
        # Check for under-provisioning (consistently high utilization)
        if avg_cpu > 75 or avg_memory > 80:
            recommendations.append(ScalingRecommendation(
                recommendation_id=f"rightsize_{service}_up_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                service=service,
                action='right_size',
                resource_type='instance_type',
                current_value=utilization_data[-1].instance_count,
                recommended_value=utilization_data[-1].instance_count * 1.3,
                urgency='high',
                lead_time_hours=4,
                cost_impact_monthly=4000,  # Estimated cost increase
                rationale=f"CPU averaging {avg_cpu:.1f}% and memory {avg_memory:.1f}% indicate under-provisioning. Consider larger instance types or more instances.",
                confidence=0.85
            ))
        
        return recommendations


class CapacityPlanningPipeline:
    """
    Production pipeline orchestrating all capacity planning components.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize capacity planning pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.demand_forecaster = DemandForecaster(self.bedrock_client)
        self.capacity_modeler = CapacityModeler(self.bedrock_client)
        self.scaling_optimizer = ScalingOptimizer(self.bedrock_client)
        
        self.insights: List[CapacityInsight] = []
        self.recommendations: List[ScalingRecommendation] = []
    
    def process_demand_data(self, data_points: List[DemandDataPoint]):
        """Process demand data points"""
        for dp in data_points:
            self.demand_forecaster.add_demand_data(dp)
    
    def process_utilization_data(self, utilization: List[ResourceUtilization]):
        """Process utilization data"""
        for u in utilization:
            self.capacity_modeler.add_utilization_data(u)
    
    def set_capacity_limits(self, limits: List[CapacityLimit]):
        """Configure capacity limits"""
        for limit in limits:
            self.capacity_modeler.set_capacity_limit(limit)
    
    def generate_capacity_plan(self,
                              service: str,
                              current_capacity_rps: float,
                              forecast_horizon_hours: int = 168) -> Dict[str, Any]:
        """
        Generate comprehensive capacity plan for a service.
        
        Args:
            service: Service to plan for
            current_capacity_rps: Current capacity in requests/second
            forecast_horizon_hours: Planning horizon
            
        Returns:
            Comprehensive capacity plan
        """
        plan = {
            'service': service,
            'generated_at': datetime.now(),
            'forecast': None,
            'growth_trend': None,
            'headroom_analysis': None,
            'bottleneck_analysis': None,
            'scaling_recommendation': None,
            'right_sizing_opportunities': []
        }
        
        # Generate demand forecast
        forecast = self.demand_forecaster.forecast_demand(service, forecast_horizon_hours)
        plan['forecast'] = forecast
        
        if forecast:
            # Analyze growth trend
            growth = self.demand_forecaster.detect_growth_trend(service)
            plan['growth_trend'] = growth
            
            # Generate scaling recommendation
            scaling_rec = self.scaling_optimizer.generate_scaling_recommendation(
                service,
                forecast,
                current_capacity_rps
            )
            plan['scaling_recommendation'] = scaling_rec
            self.recommendations.append(scaling_rec)
        
        # Analyze capacity headroom
        headroom = self.capacity_modeler.analyze_capacity_headroom(service)
        if headroom:
            plan['headroom_analysis'] = headroom
            self.insights.append(headroom)
        
        # Identify bottleneck
        bottleneck = self.capacity_modeler.identify_bottleneck(service)
        if bottleneck:
            plan['bottleneck_analysis'] = bottleneck
            self.insights.append(bottleneck)
        
        # Check for right-sizing opportunities
        if service in self.capacity_modeler.utilization_history:
            util_data = self.capacity_modeler.utilization_history[service]
            right_sizing = self.scaling_optimizer.identify_right_sizing_opportunities(
                util_data,
                service
            )
            plan['right_sizing_opportunities'] = right_sizing
            self.recommendations.extend(right_sizing)
        
        return plan
    
    def get_all_recommendations(self, min_urgency: str = 'low') -> List[ScalingRecommendation]:
        """Get all recommendations above minimum urgency"""
        urgency_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_level = urgency_order.get(min_urgency, 0)
        
        return [
            rec for rec in self.recommendations
            if urgency_order.get(rec.urgency, 0) >= min_level
        ]
    
    def get_cost_optimization_summary(self) -> Dict[str, Any]:
        """Calculate potential cost savings from recommendations"""
        scale_down_savings = sum(
            abs(rec.cost_impact_monthly)
            for rec in self.recommendations
            if rec.action == 'scale_down' or (rec.action == 'right_size' and rec.cost_impact_monthly < 0)
        )
        
        scale_up_costs = sum(
            rec.cost_impact_monthly
            for rec in self.recommendations
            if rec.action == 'scale_up' or (rec.action == 'right_size' and rec.cost_impact_monthly > 0)
        )
        
        net_savings = scale_down_savings - scale_up_costs
        
        return {
            'potential_monthly_savings': scale_down_savings,
            'required_investment': scale_up_costs,
            'net_monthly_impact': net_savings,
            'total_recommendations': len(self.recommendations),
            'scale_up_count': len([r for r in self.recommendations if r.action == 'scale_up']),
            'scale_down_count': len([r for r in self.recommendations if r.action == 'scale_down']),
            'right_size_count': len([r for r in self.recommendations if r.action == 'right_size'])
        }


# Example usage and realistic data generators
def generate_sample_demand_data(service: str,
                                base_rps: float = 1000,
                                days: int = 14) -> List[DemandDataPoint]:
    """Generate realistic demand data with daily and weekly patterns"""
    data_points = []
    start_time = datetime.now() - timedelta(days=days)
    
    for hour in range(days * 24):
        timestamp = start_time + timedelta(hours=hour)
        
        # Daily pattern (peak during business hours)
        hour_of_day = timestamp.hour
        daily_factor = 1.0 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12)
        
        # Weekly pattern (lower on weekends)
        day_of_week = timestamp.weekday()
        weekly_factor = 0.7 if day_of_week >= 5 else 1.0
        
        # Growth trend (1% per week)
        growth_factor = 1.0 + (hour / (24 * 7)) * 0.01
        
        # Combine factors with noise
        rps = base_rps * daily_factor * weekly_factor * growth_factor
        rps += np.random.normal(0, base_rps * 0.1)
        rps = max(0, rps)
        
        data_points.append(DemandDataPoint(
            timestamp=timestamp,
            service=service,
            requests_per_second=rps,
            active_users=int(rps * 50),
            transaction_volume=int(rps * 60)
        ))
    
    return data_points


def generate_sample_utilization_data(service: str,
                                    avg_utilization: float = 60,
                                    days: int = 7) -> List[ResourceUtilization]:
    """Generate realistic resource utilization data"""
    util_data = []
    start_time = datetime.now() - timedelta(days=days)
    
    for hour in range(days * 24):
        timestamp = start_time + timedelta(hours=hour)
        
        # Correlate with traffic patterns
        hour_of_day = timestamp.hour
        traffic_factor = 1.0 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12)
        
        cpu = avg_utilization * traffic_factor + np.random.normal(0, 5)
        memory = (avg_utilization + 10) * traffic_factor + np.random.normal(0, 5)
        disk = 75 + hour * 0.02 + np.random.normal(0, 2)  # Slowly filling
        
        cpu = np.clip(cpu, 0, 100)
        memory = np.clip(memory, 0, 100)
        disk = np.clip(disk, 0, 100)
        
        util_data.append(ResourceUtilization(
            timestamp=timestamp,
            service=service,
            cpu_percent=cpu,
            memory_percent=memory,
            disk_percent=disk,
            network_mbps=cpu * 2,  # Correlates with CPU
            active_connections=int(cpu * 10),
            instance_count=5
        ))
    
    return util_data


def main():
    """
    Demonstrate capacity planning pipeline with realistic scenarios.
    """
    print("=" * 80)
    print("Chapter 14: Capacity Planning with ML")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing capacity planning pipeline...")
    pipeline = CapacityPlanningPipeline(aws_region='us-east-1')
    print()
    
    # Configure capacity limits
    limits = [
        CapacityLimit('api-service', 'cpu', 80, 100, 'cores', 50),
        CapacityLimit('api-service', 'memory', 120, 150, 'GB', 30),
        CapacityLimit('api-service', 'disk', 800, 1000, 'GB', 10),
        CapacityLimit('api-service', 'network', 800, 1000, 'Mbps', 20)
    ]
    pipeline.set_capacity_limits(limits)
    print("Configured capacity limits")
    print()
    
    # Scenario 1: Demand Forecasting
    print("=" * 80)
    print("SCENARIO 1: Demand Forecasting with Growth Trend")
    print("=" * 80)
    print()
    
    # Generate demand data with growth
    print("Processing 14 days of demand history...")
    demand_data = generate_sample_demand_data('api-service', base_rps=1000, days=14)
    pipeline.process_demand_data(demand_data)
    
    # Generate forecast
    forecast = pipeline.demand_forecaster.forecast_demand('api-service', forecast_hours=168)
    
    if forecast:
        print(f"\n📈 DEMAND FORECAST")
        print(f"   Service: {forecast['service']}")
        print(f"   Current Demand: {forecast['current_demand']:.0f} req/sec")
        print(f"   Peak Demand (next week): {forecast['peak_demand']:.0f} req/sec")
        print(f"   Peak Time: {forecast['peak_time'].strftime('%Y-%m-%d %H:%M')}")
        print(f"   Growth Rate: {forecast['growth_rate_percent']:+.1f}%")
        print(f"   Forecast Confidence: {forecast['confidence']:.0%}")
    
    # Analyze growth trend
    growth = pipeline.demand_forecaster.detect_growth_trend('api-service')
    if growth:
        print(f"\n📊 GROWTH TREND ANALYSIS")
        print(f"   Growth per Day: {growth['growth_per_day']:+.1f} req/sec")
        print(f"   Growth per Week: {growth['growth_percent_per_week']:+.1f}%")
        print(f"   Current: {growth['current_demand']:.0f} req/sec")
        print(f"   Projected (1 week): {growth['projected_demand_next_week']:.0f} req/sec")
        print(f"   Significant Trend: {'Yes' if growth['is_significant'] else 'No'}")
    
    # Scenario 2: Capacity Analysis
    print("\n" + "=" * 80)
    print("SCENARIO 2: Capacity Headroom and Bottleneck Analysis")
    print("=" * 80)
    print()
    
    # Generate utilization data
    print("Processing 7 days of utilization history...")
    util_data = generate_sample_utilization_data('api-service', avg_utilization=75, days=7)
    pipeline.process_utilization_data(util_data)
    
    # Analyze headroom
    headroom = pipeline.capacity_modeler.analyze_capacity_headroom('api-service')
    if headroom:
        print(f"\n🔍 CAPACITY HEADROOM ANALYSIS")
        print(f"   Priority: {headroom.priority.upper()}")
        print(f"   CPU: {headroom.evidence['cpu_utilization']:.1f}%")
        print(f"   Memory: {headroom.evidence['memory_utilization']:.1f}%")
        print(f"   Disk: {headroom.evidence['disk_utilization']:.1f}%")
        if headroom.evidence['low_headroom_resources']:
            print(f"   ⚠️  Low Headroom: {', '.join(headroom.evidence['low_headroom_resources'])}")
        print(f"\n   Analysis:")
        print(f"   {headroom.description}")
    
    # Identify bottleneck
    bottleneck = pipeline.capacity_modeler.identify_bottleneck('api-service')
    if bottleneck:
        print(f"\n🎯 BOTTLENECK IDENTIFICATION")
        print(f"   Bottleneck: {bottleneck.evidence['bottleneck_resource']}")
        print(f"   Available Headroom: {bottleneck.evidence['bottleneck_headroom_percent']:.1f}%")
        print(f"   Priority: {bottleneck.priority.upper()}")
        print(f"\n   {bottleneck.description}")
    
    # Scenario 3: Scaling Recommendations
    print("\n" + "=" * 80)
    print("SCENARIO 3: Comprehensive Capacity Plan")
    print("=" * 80)
    print()
    
    print("Generating capacity plan...")
    plan = pipeline.generate_capacity_plan(
        'api-service',
        current_capacity_rps=1500,
        forecast_horizon_hours=168
    )
    
    if plan['scaling_recommendation']:
        rec = plan['scaling_recommendation']
        print(f"\n⚙️  SCALING RECOMMENDATION")
        print(f"   Action: {rec.action.upper().replace('_', ' ')}")
        print(f"   Current Capacity: {rec.current_value:.0f} req/sec")
        print(f"   Recommended: {rec.recommended_value:.0f} req/sec")
        print(f"   Urgency: {rec.urgency.upper()}")
        print(f"   Lead Time: {rec.lead_time_hours:.1f} hours")
        print(f"   Monthly Cost Impact: ${rec.cost_impact_monthly:,.2f}")
        print(f"   Confidence: {rec.confidence:.0%}")
        print(f"\n   Rationale:")
        print(f"   {rec.rationale}")
    
    # Right-sizing opportunities
    if plan['right_sizing_opportunities']:
        print(f"\n💰 RIGHT-SIZING OPPORTUNITIES")
        for opp in plan['right_sizing_opportunities']:
            print(f"   • {opp.rationale}")
            print(f"     Estimated Impact: ${opp.cost_impact_monthly:+,.2f}/month")
    
    # Cost optimization summary
    print("\n" + "=" * 80)
    print("COST OPTIMIZATION SUMMARY")
    print("=" * 80)
    print()
    
    cost_summary = pipeline.get_cost_optimization_summary()
    print(f"Total Recommendations: {cost_summary['total_recommendations']}")
    print(f"  Scale Up: {cost_summary['scale_up_count']}")
    print(f"  Scale Down: {cost_summary['scale_down_count']}")
    print(f"  Right-Size: {cost_summary['right_size_count']}")
    print(f"\nFinancial Impact:")
    print(f"  Potential Savings: ${cost_summary['potential_monthly_savings']:,.2f}/month")
    print(f"  Required Investment: ${cost_summary['required_investment']:,.2f}/month")
    print(f"  Net Impact: ${cost_summary['net_monthly_impact']:+,.2f}/month")
    
    print("\n" + "=" * 80)
    print("Capacity Planning Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
