"""
Chapter 18: User and Entity Behavior Analytics (UEBA)
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready UEBA including:
- Individual and peer group behavioral baselines
- Multi-dimensional behavior tracking (temporal, access, location, volume)
- Anomaly detection and risk scoring
- Compromised credential detection
- Insider threat detection
- Lateral movement detection
- Session analysis and impossible travel
- AI-powered behavioral analysis with AWS Bedrock

Author: AI-Powered Observability
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats
import math
import warnings
warnings.filterwarnings('ignore')


class EntityType(Enum):
    """Types of entities for behavior analysis"""
    USER = "user"
    SERVICE_ACCOUNT = "service_account"
    HOST = "host"
    APPLICATION = "application"


class AnomalyType(Enum):
    """Types of behavioral anomalies"""
    TEMPORAL = "temporal"
    ACCESS = "access"
    LOCATION = "location"
    VOLUME = "volume"
    ACTION = "action"
    RELATIONSHIP = "relationship"
    IMPOSSIBLE_TRAVEL = "impossible_travel"


class RiskLevel(Enum):
    """Risk levels for anomalies"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class UserActivity:
    """Represents a user activity event"""
    timestamp: datetime
    entity_id: str
    entity_type: EntityType
    action: str
    target_resource: str
    source_ip: str
    location: str  # Country/city
    latitude: float
    longitude: float
    bytes_transferred: int
    session_id: str
    device_id: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehavioralBaseline:
    """Represents an entity's behavioral baseline"""
    entity_id: str
    entity_type: EntityType
    peer_group: str
    
    # Temporal baseline
    active_hours: Set[int]  # Hours when typically active (0-23)
    active_days: Set[int]  # Days when typically active (0=Mon, 6=Sun)
    avg_session_duration_minutes: float
    avg_sessions_per_day: float
    
    # Access baseline
    typical_resources: Set[str]
    typical_actions: Set[str]
    resource_access_counts: Dict[str, int]
    
    # Location baseline
    typical_locations: Set[str]
    typical_ips: Set[str]
    typical_devices: Set[str]
    
    # Volume baseline
    avg_bytes_per_day: float
    std_bytes_per_day: float
    avg_actions_per_day: float
    std_actions_per_day: float
    
    # Relationship baseline
    typical_collaborators: Set[str]
    
    # Metadata
    last_updated: datetime
    data_points: int


@dataclass
class BehaviorAnomaly:
    """Represents a detected behavioral anomaly"""
    anomaly_id: str
    timestamp: datetime
    entity_id: str
    entity_type: EntityType
    anomaly_type: AnomalyType
    description: str
    risk_score: float
    risk_level: RiskLevel
    evidence: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    recommended_actions: List[str]
    mitre_techniques: List[str]


@dataclass
class RiskScore:
    """Composite risk score for an entity"""
    entity_id: str
    timestamp: datetime
    overall_score: float
    component_scores: Dict[str, float]
    active_anomalies: List[BehaviorAnomaly]
    risk_factors: List[str]
    trend: str  # increasing, stable, decreasing


class BehavioralBaselineBuilder:
    """
    Builds and maintains behavioral baselines for entities.
    """
    
    def __init__(self, learning_period_days: int = 30):
        """
        Initialize baseline builder.
        
        Args:
            learning_period_days: Days of history to consider for baselines
        """
        self.learning_period_days = learning_period_days
        self.baselines: Dict[str, BehavioralBaseline] = {}
        self.activity_history: Dict[str, List[UserActivity]] = defaultdict(list)
        self.peer_groups: Dict[str, Set[str]] = defaultdict(set)  # group_name -> entity_ids
    
    def add_activity(self, activity: UserActivity):
        """Add activity to history for baseline building"""
        self.activity_history[activity.entity_id].append(activity)
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(days=self.learning_period_days)
        self.activity_history[activity.entity_id] = [
            a for a in self.activity_history[activity.entity_id]
            if a.timestamp > cutoff
        ]
    
    def assign_peer_group(self, entity_id: str, peer_group: str):
        """Assign entity to a peer group"""
        self.peer_groups[peer_group].add(entity_id)
    
    def build_baseline(self, entity_id: str, entity_type: EntityType, peer_group: str = 'default') -> BehavioralBaseline:
        """
        Build behavioral baseline for an entity.
        
        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            peer_group: Peer group name
            
        Returns:
            Behavioral baseline
        """
        activities = self.activity_history.get(entity_id, [])
        
        if not activities:
            # Return empty baseline for new entities
            return self._create_empty_baseline(entity_id, entity_type, peer_group)
        
        # Temporal analysis
        active_hours = set(a.timestamp.hour for a in activities)
        active_days = set(a.timestamp.weekday() for a in activities)
        
        # Calculate session durations
        sessions = defaultdict(list)
        for activity in activities:
            sessions[activity.session_id].append(activity.timestamp)
        
        session_durations = []
        for session_times in sessions.values():
            if len(session_times) >= 2:
                duration = (max(session_times) - min(session_times)).total_seconds() / 60
                session_durations.append(duration)
        
        avg_session_duration = np.mean(session_durations) if session_durations else 30.0
        
        # Sessions per day
        days_active = len(set(a.timestamp.date() for a in activities))
        unique_sessions = len(set(a.session_id for a in activities))
        avg_sessions_per_day = unique_sessions / max(days_active, 1)
        
        # Access patterns
        typical_resources = set(a.target_resource for a in activities)
        typical_actions = set(a.action for a in activities)
        resource_counts = Counter(a.target_resource for a in activities)
        
        # Location patterns
        typical_locations = set(a.location for a in activities)
        typical_ips = set(a.source_ip for a in activities)
        typical_devices = set(a.device_id for a in activities)
        
        # Volume patterns
        daily_bytes = defaultdict(int)
        daily_actions = defaultdict(int)
        for activity in activities:
            day_key = activity.timestamp.date()
            daily_bytes[day_key] += activity.bytes_transferred
            daily_actions[day_key] += 1
        
        bytes_values = list(daily_bytes.values()) or [0]
        action_values = list(daily_actions.values()) or [0]
        
        # Relationship patterns (from metadata or resource access)
        collaborators = set()
        for activity in activities:
            if 'collaborator' in activity.metadata:
                collaborators.add(activity.metadata['collaborator'])
        
        baseline = BehavioralBaseline(
            entity_id=entity_id,
            entity_type=entity_type,
            peer_group=peer_group,
            active_hours=active_hours,
            active_days=active_days,
            avg_session_duration_minutes=avg_session_duration,
            avg_sessions_per_day=avg_sessions_per_day,
            typical_resources=typical_resources,
            typical_actions=typical_actions,
            resource_access_counts=dict(resource_counts),
            typical_locations=typical_locations,
            typical_ips=typical_ips,
            typical_devices=typical_devices,
            avg_bytes_per_day=np.mean(bytes_values),
            std_bytes_per_day=np.std(bytes_values) if len(bytes_values) > 1 else np.mean(bytes_values) * 0.5,
            avg_actions_per_day=np.mean(action_values),
            std_actions_per_day=np.std(action_values) if len(action_values) > 1 else np.mean(action_values) * 0.5,
            typical_collaborators=collaborators,
            last_updated=datetime.now(),
            data_points=len(activities)
        )
        
        self.baselines[entity_id] = baseline
        return baseline
    
    def _create_empty_baseline(self, entity_id: str, entity_type: EntityType, peer_group: str) -> BehavioralBaseline:
        """Create empty baseline for new entity"""
        return BehavioralBaseline(
            entity_id=entity_id,
            entity_type=entity_type,
            peer_group=peer_group,
            active_hours=set(range(9, 18)),  # Default business hours
            active_days=set(range(0, 5)),  # Weekdays
            avg_session_duration_minutes=30.0,
            avg_sessions_per_day=2.0,
            typical_resources=set(),
            typical_actions=set(),
            resource_access_counts={},
            typical_locations=set(),
            typical_ips=set(),
            typical_devices=set(),
            avg_bytes_per_day=0,
            std_bytes_per_day=1000000,
            avg_actions_per_day=10,
            std_actions_per_day=5,
            typical_collaborators=set(),
            last_updated=datetime.now(),
            data_points=0
        )
    
    def build_peer_baseline(self, peer_group: str) -> BehavioralBaseline:
        """
        Build aggregate baseline for a peer group.
        
        Args:
            peer_group: Peer group name
            
        Returns:
            Aggregate peer baseline
        """
        entity_ids = self.peer_groups.get(peer_group, set())
        
        if not entity_ids:
            return self._create_empty_baseline(f"peer_{peer_group}", EntityType.USER, peer_group)
        
        # Aggregate all activities from peer group
        all_activities = []
        for entity_id in entity_ids:
            all_activities.extend(self.activity_history.get(entity_id, []))
        
        if not all_activities:
            return self._create_empty_baseline(f"peer_{peer_group}", EntityType.USER, peer_group)
        
        # Build aggregate baseline
        active_hours = set(a.timestamp.hour for a in all_activities)
        active_days = set(a.timestamp.weekday() for a in all_activities)
        typical_resources = set(a.target_resource for a in all_activities)
        typical_actions = set(a.action for a in all_activities)
        typical_locations = set(a.location for a in all_activities)
        
        # Aggregate volumes per entity
        entity_daily_bytes = defaultdict(lambda: defaultdict(int))
        for activity in all_activities:
            day_key = activity.timestamp.date()
            entity_daily_bytes[activity.entity_id][day_key] += activity.bytes_transferred
        
        all_daily_bytes = []
        for entity_bytes in entity_daily_bytes.values():
            all_daily_bytes.extend(entity_bytes.values())
        
        return BehavioralBaseline(
            entity_id=f"peer_{peer_group}",
            entity_type=EntityType.USER,
            peer_group=peer_group,
            active_hours=active_hours,
            active_days=active_days,
            avg_session_duration_minutes=30.0,
            avg_sessions_per_day=3.0,
            typical_resources=typical_resources,
            typical_actions=typical_actions,
            resource_access_counts={},
            typical_locations=typical_locations,
            typical_ips=set(),
            typical_devices=set(),
            avg_bytes_per_day=np.mean(all_daily_bytes) if all_daily_bytes else 0,
            std_bytes_per_day=np.std(all_daily_bytes) if len(all_daily_bytes) > 1 else 1000000,
            avg_actions_per_day=len(all_activities) / max(len(entity_ids), 1) / 30,
            std_actions_per_day=5,
            typical_collaborators=set(),
            last_updated=datetime.now(),
            data_points=len(all_activities)
        )
    
    def get_baseline(self, entity_id: str) -> Optional[BehavioralBaseline]:
        """Get baseline for an entity"""
        return self.baselines.get(entity_id)


class BehaviorAnomalyDetector:
    """
    Detects behavioral anomalies by comparing activity against baselines.
    """
    
    def __init__(self, bedrock_client, baseline_builder: BehavioralBaselineBuilder):
        """Initialize anomaly detector"""
        self.bedrock_client = bedrock_client
        self.baseline_builder = baseline_builder
        self.recent_activities: Dict[str, List[UserActivity]] = defaultdict(list)
    
    def detect_anomalies(self, activity: UserActivity) -> List[BehaviorAnomaly]:
        """
        Detect anomalies in user activity.
        
        Args:
            activity: User activity to analyze
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Get entity baseline
        baseline = self.baseline_builder.get_baseline(activity.entity_id)
        
        if not baseline:
            # Build baseline if doesn't exist
            baseline = self.baseline_builder.build_baseline(
                activity.entity_id,
                activity.entity_type,
                'default'
            )
        
        # Get peer baseline
        peer_baseline = self.baseline_builder.build_peer_baseline(baseline.peer_group)
        
        # Check temporal anomalies
        temporal_anomaly = self._check_temporal_anomaly(activity, baseline)
        if temporal_anomaly:
            anomalies.append(temporal_anomaly)
        
        # Check access anomalies
        access_anomaly = self._check_access_anomaly(activity, baseline, peer_baseline)
        if access_anomaly:
            anomalies.append(access_anomaly)
        
        # Check location anomalies
        location_anomaly = self._check_location_anomaly(activity, baseline)
        if location_anomaly:
            anomalies.append(location_anomaly)
        
        # Check impossible travel
        travel_anomaly = self._check_impossible_travel(activity)
        if travel_anomaly:
            anomalies.append(travel_anomaly)
        
        # Check volume anomalies
        volume_anomaly = self._check_volume_anomaly(activity, baseline)
        if volume_anomaly:
            anomalies.append(volume_anomaly)
        
        # Track recent activities for pattern analysis
        self.recent_activities[activity.entity_id].append(activity)
        # Keep last 100 activities per entity
        self.recent_activities[activity.entity_id] = self.recent_activities[activity.entity_id][-100:]
        
        return anomalies
    
    def _check_temporal_anomaly(self,
                               activity: UserActivity,
                               baseline: BehavioralBaseline) -> Optional[BehaviorAnomaly]:
        """Check for temporal anomalies"""
        hour = activity.timestamp.hour
        day = activity.timestamp.weekday()
        
        hour_anomaly = hour not in baseline.active_hours if baseline.active_hours else False
        day_anomaly = day not in baseline.active_days if baseline.active_days else False
        
        if hour_anomaly or day_anomaly:
            anomaly_details = []
            if hour_anomaly:
                anomaly_details.append(f"Activity at hour {hour} (typical: {sorted(baseline.active_hours)})")
            if day_anomaly:
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                anomaly_details.append(f"Activity on {day_names[day]} (atypical day)")
            
            risk_score = 0.5 if hour_anomaly else 0
            risk_score += 0.3 if day_anomaly else 0
            
            # Higher risk for very unusual hours
            if hour in [0, 1, 2, 3, 4, 5]:
                risk_score += 0.3
            
            return BehaviorAnomaly(
                anomaly_id=f"temporal_{activity.entity_id}_{activity.timestamp.timestamp()}",
                timestamp=activity.timestamp,
                entity_id=activity.entity_id,
                entity_type=activity.entity_type,
                anomaly_type=AnomalyType.TEMPORAL,
                description=f"Activity at unusual time: {'; '.join(anomaly_details)}",
                risk_score=min(risk_score, 1.0),
                risk_level=self._score_to_level(risk_score),
                evidence={
                    'activity_hour': hour,
                    'activity_day': day,
                    'typical_hours': list(baseline.active_hours),
                    'typical_days': list(baseline.active_days)
                },
                baseline_comparison={
                    'hour_in_baseline': hour in baseline.active_hours,
                    'day_in_baseline': day in baseline.active_days
                },
                recommended_actions=[
                    "Verify if user is traveling or working remotely",
                    "Check for other anomalous activity in this session",
                    "Contact user to confirm legitimate access"
                ],
                mitre_techniques=["T1078 - Valid Accounts"]
            )
        
        return None
    
    def _check_access_anomaly(self,
                             activity: UserActivity,
                             baseline: BehavioralBaseline,
                             peer_baseline: BehavioralBaseline) -> Optional[BehaviorAnomaly]:
        """Check for access pattern anomalies"""
        resource = activity.target_resource
        action = activity.action
        
        # Check if resource is new for user
        new_resource = resource not in baseline.typical_resources if baseline.typical_resources else False
        # Check if resource is unusual for peer group
        unusual_for_peers = resource not in peer_baseline.typical_resources if peer_baseline.typical_resources else False
        
        # Check if action is new for user
        new_action = action not in baseline.typical_actions if baseline.typical_actions else False
        
        if new_resource or (new_resource and unusual_for_peers):
            risk_score = 0.4 if new_resource else 0
            risk_score += 0.3 if unusual_for_peers else 0
            risk_score += 0.2 if new_action else 0
            
            # Boost risk for sensitive resources
            if any(kw in resource.lower() for kw in ['admin', 'finance', 'hr', 'secret', 'confidential']):
                risk_score += 0.3
            
            description_parts = []
            if new_resource:
                description_parts.append(f"First-time access to {resource}")
            if unusual_for_peers:
                description_parts.append(f"Resource unusual for peer group")
            if new_action:
                description_parts.append(f"New action type: {action}")
            
            return BehaviorAnomaly(
                anomaly_id=f"access_{activity.entity_id}_{activity.timestamp.timestamp()}",
                timestamp=activity.timestamp,
                entity_id=activity.entity_id,
                entity_type=activity.entity_type,
                anomaly_type=AnomalyType.ACCESS,
                description="; ".join(description_parts),
                risk_score=min(risk_score, 1.0),
                risk_level=self._score_to_level(risk_score),
                evidence={
                    'resource': resource,
                    'action': action,
                    'user_typical_resources': len(baseline.typical_resources),
                    'peer_typical_resources': len(peer_baseline.typical_resources)
                },
                baseline_comparison={
                    'new_for_user': new_resource,
                    'unusual_for_peers': unusual_for_peers,
                    'new_action': new_action
                },
                recommended_actions=[
                    f"Review access justification for {resource}",
                    "Check if user role has changed",
                    "Verify with user's manager if access is appropriate"
                ],
                mitre_techniques=["T1078 - Valid Accounts", "T1021 - Remote Services"]
            )
        
        return None
    
    def _check_location_anomaly(self,
                               activity: UserActivity,
                               baseline: BehavioralBaseline) -> Optional[BehaviorAnomaly]:
        """Check for location-based anomalies"""
        new_location = activity.location not in baseline.typical_locations if baseline.typical_locations else False
        new_ip = activity.source_ip not in baseline.typical_ips if baseline.typical_ips else False
        new_device = activity.device_id not in baseline.typical_devices if baseline.typical_devices else False
        
        if new_location or (new_ip and new_device):
            risk_score = 0.5 if new_location else 0
            risk_score += 0.2 if new_ip else 0
            risk_score += 0.2 if new_device else 0
            
            # High-risk countries boost score
            high_risk_countries = ['Russia', 'China', 'North Korea', 'Iran']
            if any(country.lower() in activity.location.lower() for country in high_risk_countries):
                risk_score += 0.4
            
            description_parts = []
            if new_location:
                description_parts.append(f"New location: {activity.location}")
            if new_ip:
                description_parts.append(f"New IP: {activity.source_ip}")
            if new_device:
                description_parts.append(f"New device: {activity.device_id}")
            
            return BehaviorAnomaly(
                anomaly_id=f"location_{activity.entity_id}_{activity.timestamp.timestamp()}",
                timestamp=activity.timestamp,
                entity_id=activity.entity_id,
                entity_type=activity.entity_type,
                anomaly_type=AnomalyType.LOCATION,
                description="; ".join(description_parts),
                risk_score=min(risk_score, 1.0),
                risk_level=self._score_to_level(risk_score),
                evidence={
                    'current_location': activity.location,
                    'current_ip': activity.source_ip,
                    'current_device': activity.device_id,
                    'typical_locations': list(baseline.typical_locations)
                },
                baseline_comparison={
                    'new_location': new_location,
                    'new_ip': new_ip,
                    'new_device': new_device
                },
                recommended_actions=[
                    "Verify user's current location (travel, remote work)",
                    "Require MFA re-authentication",
                    "Check for impossible travel with other logins"
                ],
                mitre_techniques=["T1078 - Valid Accounts", "T1133 - External Remote Services"]
            )
        
        return None
    
    def _check_impossible_travel(self, activity: UserActivity) -> Optional[BehaviorAnomaly]:
        """Check for impossible travel (logins from distant locations in short time)"""
        recent = self.recent_activities.get(activity.entity_id, [])
        
        if not recent:
            return None
        
        # Check last activity
        last_activity = recent[-1] if recent else None
        
        if not last_activity or last_activity.location == activity.location:
            return None
        
        # Calculate time and distance
        time_diff_hours = (activity.timestamp - last_activity.timestamp).total_seconds() / 3600
        
        # Calculate distance using Haversine formula
        distance_km = self._haversine_distance(
            last_activity.latitude, last_activity.longitude,
            activity.latitude, activity.longitude
        )
        
        # Assume max travel speed of 1000 km/h (fast plane)
        max_possible_distance = time_diff_hours * 1000
        
        if distance_km > max_possible_distance and time_diff_hours < 24:
            return BehaviorAnomaly(
                anomaly_id=f"travel_{activity.entity_id}_{activity.timestamp.timestamp()}",
                timestamp=activity.timestamp,
                entity_id=activity.entity_id,
                entity_type=activity.entity_type,
                anomaly_type=AnomalyType.IMPOSSIBLE_TRAVEL,
                description=f"Impossible travel detected: {distance_km:.0f}km in {time_diff_hours:.1f} hours",
                risk_score=0.95,
                risk_level=RiskLevel.CRITICAL,
                evidence={
                    'previous_location': last_activity.location,
                    'current_location': activity.location,
                    'distance_km': distance_km,
                    'time_diff_hours': time_diff_hours,
                    'max_possible_km': max_possible_distance
                },
                baseline_comparison={
                    'physically_impossible': True
                },
                recommended_actions=[
                    "Immediately lock the account",
                    "Contact user via alternate channel",
                    "Investigate both sessions for malicious activity",
                    "Check for credential compromise"
                ],
                mitre_techniques=["T1078 - Valid Accounts", "T1110 - Brute Force"]
            )
        
        return None
    
    def _check_volume_anomaly(self,
                             activity: UserActivity,
                             baseline: BehavioralBaseline) -> Optional[BehaviorAnomaly]:
        """Check for data volume anomalies"""
        if baseline.std_bytes_per_day == 0:
            return None
        
        # Check if this single activity's bytes are anomalous
        bytes_zscore = (activity.bytes_transferred - baseline.avg_bytes_per_day / 10) / (baseline.std_bytes_per_day / 10 + 1)
        
        if bytes_zscore > 3:  # More than 3 standard deviations
            risk_score = min(0.3 + (bytes_zscore - 3) * 0.1, 1.0)
            
            return BehaviorAnomaly(
                anomaly_id=f"volume_{activity.entity_id}_{activity.timestamp.timestamp()}",
                timestamp=activity.timestamp,
                entity_id=activity.entity_id,
                entity_type=activity.entity_type,
                anomaly_type=AnomalyType.VOLUME,
                description=f"Unusual data volume: {activity.bytes_transferred / 1_000_000:.1f}MB transferred",
                risk_score=risk_score,
                risk_level=self._score_to_level(risk_score),
                evidence={
                    'bytes_transferred': activity.bytes_transferred,
                    'baseline_avg': baseline.avg_bytes_per_day,
                    'baseline_std': baseline.std_bytes_per_day,
                    'z_score': bytes_zscore
                },
                baseline_comparison={
                    'above_normal': True,
                    'standard_deviations': bytes_zscore
                },
                recommended_actions=[
                    "Review what data was accessed/transferred",
                    "Check for data exfiltration indicators",
                    "Verify business justification for large transfer"
                ],
                mitre_techniques=["T1048 - Exfiltration Over Alternative Protocol"]
            )
        
        return None
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level"""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        elif score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFO


class EntityRiskScorer:
    """
    Calculates and tracks risk scores for entities.
    """
    
    def __init__(self, bedrock_client):
        """Initialize risk scorer"""
        self.bedrock_client = bedrock_client
        self.entity_anomalies: Dict[str, List[BehaviorAnomaly]] = defaultdict(list)
        self.entity_scores: Dict[str, List[RiskScore]] = defaultdict(list)
    
    def add_anomaly(self, anomaly: BehaviorAnomaly):
        """Add anomaly to entity's history"""
        self.entity_anomalies[anomaly.entity_id].append(anomaly)
        
        # Keep last 30 days of anomalies
        cutoff = datetime.now() - timedelta(days=30)
        self.entity_anomalies[anomaly.entity_id] = [
            a for a in self.entity_anomalies[anomaly.entity_id]
            if a.timestamp > cutoff
        ]
    
    def calculate_risk_score(self, entity_id: str) -> RiskScore:
        """
        Calculate comprehensive risk score for an entity.
        
        Args:
            entity_id: Entity to score
            
        Returns:
            Risk score with components
        """
        anomalies = self.entity_anomalies.get(entity_id, [])
        
        # Component scores
        component_scores = {
            'recent_anomaly_count': 0,
            'anomaly_severity': 0,
            'anomaly_diversity': 0,
            'trend': 0
        }
        
        risk_factors = []
        
        if not anomalies:
            return RiskScore(
                entity_id=entity_id,
                timestamp=datetime.now(),
                overall_score=0,
                component_scores=component_scores,
                active_anomalies=[],
                risk_factors=[],
                trend='stable'
            )
        
        # Recent anomaly count (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_anomalies = [a for a in anomalies if a.timestamp > recent_cutoff]
        component_scores['recent_anomaly_count'] = min(len(recent_anomalies) * 0.15, 0.5)
        
        if len(recent_anomalies) >= 3:
            risk_factors.append(f"{len(recent_anomalies)} anomalies in last 24 hours")
        
        # Average anomaly severity
        if anomalies:
            avg_severity = np.mean([a.risk_score for a in anomalies])
            component_scores['anomaly_severity'] = avg_severity * 0.4
            
            if avg_severity > 0.7:
                risk_factors.append("High severity anomalies detected")
        
        # Anomaly type diversity (multiple types = more concerning)
        anomaly_types = set(a.anomaly_type for a in anomalies)
        component_scores['anomaly_diversity'] = min(len(anomaly_types) * 0.1, 0.3)
        
        if len(anomaly_types) >= 3:
            risk_factors.append(f"Multiple anomaly types: {[t.value for t in anomaly_types]}")
        
        # Trend analysis
        if len(self.entity_scores.get(entity_id, [])) >= 2:
            prev_scores = [s.overall_score for s in self.entity_scores[entity_id][-5:]]
            if len(prev_scores) >= 2:
                trend_direction = prev_scores[-1] - prev_scores[0]
                if trend_direction > 0.1:
                    trend = 'increasing'
                    component_scores['trend'] = 0.1
                    risk_factors.append("Risk trend increasing")
                elif trend_direction < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Calculate overall score
        overall_score = sum(component_scores.values())
        overall_score = min(max(overall_score, 0), 1)
        
        # Check for critical indicators
        if any(a.anomaly_type == AnomalyType.IMPOSSIBLE_TRAVEL for a in recent_anomalies):
            overall_score = max(overall_score, 0.9)
            risk_factors.append("Impossible travel detected")
        
        risk_score = RiskScore(
            entity_id=entity_id,
            timestamp=datetime.now(),
            overall_score=overall_score,
            component_scores=component_scores,
            active_anomalies=recent_anomalies,
            risk_factors=risk_factors,
            trend=trend
        )
        
        self.entity_scores[entity_id].append(risk_score)
        
        return risk_score
    
    def get_high_risk_entities(self, threshold: float = 0.6) -> List[Tuple[str, RiskScore]]:
        """Get entities with risk score above threshold"""
        high_risk = []
        
        for entity_id in self.entity_anomalies.keys():
            score = self.calculate_risk_score(entity_id)
            if score.overall_score >= threshold:
                high_risk.append((entity_id, score))
        
        return sorted(high_risk, key=lambda x: x[1].overall_score, reverse=True)


class UEBAPipeline:
    """
    Production UEBA pipeline orchestrating all components.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize UEBA pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.baseline_builder = BehavioralBaselineBuilder(learning_period_days=30)
        self.anomaly_detector = BehaviorAnomalyDetector(self.bedrock_client, self.baseline_builder)
        self.risk_scorer = EntityRiskScorer(self.bedrock_client)
        
        self.all_anomalies: List[BehaviorAnomaly] = []
    
    def configure_peer_groups(self, peer_assignments: Dict[str, str]):
        """
        Configure peer group assignments.
        
        Args:
            peer_assignments: Dict of entity_id -> peer_group
        """
        for entity_id, peer_group in peer_assignments.items():
            self.baseline_builder.assign_peer_group(entity_id, peer_group)
    
    def train_baselines(self, historical_activities: List[UserActivity]):
        """
        Train baselines from historical activity data.
        
        Args:
            historical_activities: Historical activities for training
        """
        # Add all historical activities
        for activity in historical_activities:
            self.baseline_builder.add_activity(activity)
        
        # Build baselines for all entities
        entity_ids = set(a.entity_id for a in historical_activities)
        
        for entity_id in entity_ids:
            entity_type = historical_activities[0].entity_type
            peer_group = 'default'
            
            # Find peer group if assigned
            for group, members in self.baseline_builder.peer_groups.items():
                if entity_id in members:
                    peer_group = group
                    break
            
            self.baseline_builder.build_baseline(entity_id, entity_type, peer_group)
    
    def process_activity(self, activity: UserActivity) -> List[BehaviorAnomaly]:
        """
        Process a new activity and detect anomalies.
        
        Args:
            activity: New user activity
            
        Returns:
            List of detected anomalies
        """
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(activity)
        
        # Add anomalies to risk scorer
        for anomaly in anomalies:
            self.risk_scorer.add_anomaly(anomaly)
        
        # Add to history
        self.baseline_builder.add_activity(activity)
        
        self.all_anomalies.extend(anomalies)
        
        return anomalies
    
    def get_entity_risk(self, entity_id: str) -> RiskScore:
        """Get current risk score for an entity"""
        return self.risk_scorer.calculate_risk_score(entity_id)
    
    def get_high_risk_entities(self, threshold: float = 0.5) -> List[Tuple[str, RiskScore]]:
        """Get all high-risk entities"""
        return self.risk_scorer.get_high_risk_entities(threshold)
    
    def generate_ueba_report(self, entity_id: str) -> str:
        """Generate AI-powered UEBA report for an entity"""
        baseline = self.baseline_builder.get_baseline(entity_id)
        risk_score = self.risk_scorer.calculate_risk_score(entity_id)
        recent_anomalies = self.risk_scorer.entity_anomalies.get(entity_id, [])[-10:]
        
        baseline_text = "No baseline established" if not baseline else f"""
Typical Active Hours: {sorted(baseline.active_hours) if baseline.active_hours else 'Unknown'}
Typical Locations: {list(baseline.typical_locations)[:3] if baseline.typical_locations else 'Unknown'}
Typical Resources: {len(baseline.typical_resources)} resources
Data Points: {baseline.data_points}"""
        
        anomalies_text = "\n".join([
            f"- [{a.timestamp}] {a.anomaly_type.value}: {a.description}"
            for a in recent_anomalies[:5]
        ]) if recent_anomalies else "No recent anomalies"
        
        prompt = f"""Generate a UEBA (User and Entity Behavior Analytics) security report:

Entity: {entity_id}
Risk Score: {risk_score.overall_score:.2f}
Risk Level: {risk_score.risk_factors}
Trend: {risk_score.trend}

Baseline Profile:{baseline_text}

Recent Anomalies:
{anomalies_text}

Provide a security assessment covering:
1. Overall risk assessment (2-3 sentences)
2. Key behavioral concerns
3. Recommended investigation steps
4. Suggested security actions

Be specific and actionable for a security analyst."""

        try:
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 500,
                    'messages': [{
                        'role': 'user',
                        'content': prompt
                    }]
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
        
        except Exception as e:
            return f"Report generation failed: {e}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get UEBA statistics"""
        anomaly_counts = Counter(a.anomaly_type.value for a in self.all_anomalies)
        severity_counts = Counter(a.risk_level.value for a in self.all_anomalies)
        
        return {
            'total_anomalies': len(self.all_anomalies),
            'entities_monitored': len(self.baseline_builder.baselines),
            'anomalies_by_type': dict(anomaly_counts),
            'anomalies_by_severity': dict(severity_counts),
            'peer_groups': len(self.baseline_builder.peer_groups)
        }


# Example usage and realistic data generators
def generate_normal_activity(entity_id: str, days: int = 30) -> List[UserActivity]:
    """Generate normal user activity for baseline training"""
    activities = []
    base_time = datetime.now() - timedelta(days=days)
    
    # Normal working hours, typical resources
    typical_resources = ['email', 'sharepoint', 'confluence', 'jira', 'github']
    typical_actions = ['read', 'write', 'login', 'logout']
    
    for day in range(days):
        # 5-15 activities per day during business hours
        num_activities = np.random.randint(5, 15)
        
        for _ in range(num_activities):
            hour = np.random.choice(range(9, 18))  # Business hours
            minute = np.random.randint(0, 60)
            
            activity_time = base_time + timedelta(days=day, hours=hour, minutes=minute)
            
            # Skip weekends occasionally
            if activity_time.weekday() >= 5 and np.random.random() > 0.1:
                continue
            
            activities.append(UserActivity(
                timestamp=activity_time,
                entity_id=entity_id,
                entity_type=EntityType.USER,
                action=np.random.choice(typical_actions),
                target_resource=np.random.choice(typical_resources),
                source_ip='192.168.1.100',
                location='New York, USA',
                latitude=40.7128,
                longitude=-74.0060,
                bytes_transferred=np.random.randint(1000, 100000),
                session_id=f"session_{day}_{hour}",
                device_id='device_001',
                success=True
            ))
    
    return activities


def generate_anomalous_activity(entity_id: str) -> List[UserActivity]:
    """Generate anomalous activity simulating account compromise"""
    activities = []
    base_time = datetime.now()
    
    # Anomalous: Login from different country at unusual hour
    activities.append(UserActivity(
        timestamp=base_time,
        entity_id=entity_id,
        entity_type=EntityType.USER,
        action='login',
        target_resource='vpn',
        source_ip='185.143.223.50',
        location='Moscow, Russia',
        latitude=55.7558,
        longitude=37.6173,
        bytes_transferred=5000,
        session_id='suspicious_session_001',
        device_id='unknown_device',
        success=True
    ))
    
    # Anomalous: Accessing sensitive resources never accessed before
    activities.append(UserActivity(
        timestamp=base_time + timedelta(minutes=5),
        entity_id=entity_id,
        entity_type=EntityType.USER,
        action='read',
        target_resource='hr_salaries_database',
        source_ip='185.143.223.50',
        location='Moscow, Russia',
        latitude=55.7558,
        longitude=37.6173,
        bytes_transferred=50_000_000,  # Large download
        session_id='suspicious_session_001',
        device_id='unknown_device',
        success=True
    ))
    
    # Anomalous: Large data transfer
    activities.append(UserActivity(
        timestamp=base_time + timedelta(minutes=10),
        entity_id=entity_id,
        entity_type=EntityType.USER,
        action='download',
        target_resource='confidential_documents',
        source_ip='185.143.223.50',
        location='Moscow, Russia',
        latitude=55.7558,
        longitude=37.6173,
        bytes_transferred=200_000_000,  # 200MB
        session_id='suspicious_session_001',
        device_id='unknown_device',
        success=True
    ))
    
    return activities


def main():
    """
    Demonstrate UEBA pipeline with realistic scenarios.
    """
    print("=" * 80)
    print("Chapter 18: User and Entity Behavior Analytics (UEBA)")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing UEBA pipeline...")
    pipeline = UEBAPipeline(aws_region='us-east-1')
    print("✓ Pipeline initialized")
    print()
    
    # Configure peer groups
    peer_assignments = {
        'jsmith': 'engineering',
        'ajones': 'engineering',
        'bwilson': 'finance',
        'mgarcia': 'finance'
    }
    pipeline.configure_peer_groups(peer_assignments)
    print(f"✓ Configured {len(set(peer_assignments.values()))} peer groups")
    print()
    
    # Generate and train on normal activity
    print("=" * 80)
    print("PHASE 1: Building Behavioral Baselines")
    print("=" * 80)
    print()
    
    print("Generating 30 days of normal activity for training...")
    normal_activities = []
    for entity_id in peer_assignments.keys():
        activities = generate_normal_activity(entity_id, days=30)
        normal_activities.extend(activities)
    
    print(f"Generated {len(normal_activities)} normal activities")
    
    print("Training baselines...")
    pipeline.train_baselines(normal_activities)
    
    # Show baseline for one user
    baseline = pipeline.baseline_builder.get_baseline('jsmith')
    if baseline:
        print(f"\nSample Baseline for 'jsmith':")
        print(f"  Active Hours: {sorted(baseline.active_hours)}")
        print(f"  Active Days: {sorted(baseline.active_days)} (0=Mon)")
        print(f"  Typical Resources: {list(baseline.typical_resources)[:5]}")
        print(f"  Typical Locations: {list(baseline.typical_locations)}")
        print(f"  Avg Bytes/Day: {baseline.avg_bytes_per_day:,.0f}")
        print(f"  Data Points: {baseline.data_points}")
    
    # Process anomalous activity
    print("\n" + "=" * 80)
    print("PHASE 2: Detecting Behavioral Anomalies")
    print("=" * 80)
    print()
    
    print("Simulating account compromise for 'jsmith'...")
    anomalous_activities = generate_anomalous_activity('jsmith')
    
    all_anomalies = []
    for activity in anomalous_activities:
        print(f"\nProcessing: {activity.action} on {activity.target_resource} from {activity.location}")
        anomalies = pipeline.process_activity(activity)
        all_anomalies.extend(anomalies)
        
        for anomaly in anomalies:
            level_icon = "🔴" if anomaly.risk_level == RiskLevel.CRITICAL else "🟠" if anomaly.risk_level == RiskLevel.HIGH else "🟡"
            print(f"  {level_icon} {anomaly.anomaly_type.value}: {anomaly.description}")
            print(f"      Risk Score: {anomaly.risk_score:.2f}")
    
    # Get entity risk score
    print("\n" + "=" * 80)
    print("PHASE 3: Entity Risk Assessment")
    print("=" * 80)
    print()
    
    risk_score = pipeline.get_entity_risk('jsmith')
    
    print(f"Entity: jsmith")
    print(f"Overall Risk Score: {risk_score.overall_score:.2f}")
    print(f"Trend: {risk_score.trend}")
    print(f"\nComponent Scores:")
    for component, score in risk_score.component_scores.items():
        print(f"  {component}: {score:.3f}")
    
    print(f"\nRisk Factors:")
    for factor in risk_score.risk_factors:
        print(f"  ⚠ {factor}")
    
    # Get all high-risk entities
    print("\n" + "=" * 80)
    print("HIGH-RISK ENTITIES")
    print("=" * 80)
    print()
    
    high_risk = pipeline.get_high_risk_entities(threshold=0.3)
    
    if high_risk:
        for entity_id, score in high_risk:
            print(f"  {entity_id}: {score.overall_score:.2f} ({score.trend})")
    else:
        print("  No high-risk entities detected")
    
    # Generate UEBA report
    print("\n" + "=" * 80)
    print("UEBA SECURITY REPORT")
    print("=" * 80)
    print()
    
    report = pipeline.generate_ueba_report('jsmith')
    print(report)
    
    # Statistics
    print("\n" + "=" * 80)
    print("UEBA STATISTICS")
    print("=" * 80)
    print()
    
    stats = pipeline.get_statistics()
    print(f"Total Anomalies Detected: {stats['total_anomalies']}")
    print(f"Entities Monitored: {stats['entities_monitored']}")
    print(f"Peer Groups: {stats['peer_groups']}")
    
    print(f"\nAnomalies by Type:")
    for anomaly_type, count in stats['anomalies_by_type'].items():
        print(f"  {anomaly_type}: {count}")
    
    print(f"\nAnomalies by Severity:")
    for severity, count in stats['anomalies_by_severity'].items():
        print(f"  {severity}: {count}")
    
    print("\n" + "=" * 80)
    print("UEBA Demonstration Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
