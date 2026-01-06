"""
Chapter 9: Behavioral Anomaly Detection
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready behavioral anomaly detection including:
- User behavior analytics (UBA/UEBA)
- Request and transaction pattern detection
- System behavior modeling
- Graph-based anomaly detection
- Integration with AWS Bedrock for intelligent analysis

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
class UserEvent:
    """Represents a user action or event"""
    user_id: str
    timestamp: datetime
    action: str
    resource: str
    source_ip: str
    success: bool
    metadata: Dict[str, Any]


@dataclass
class UserProfile:
    """Profile capturing normal user behavior"""
    user_id: str
    typical_hours: List[int]  # Hours of day typically active
    typical_resources: Dict[str, int]  # Resources accessed and frequency
    typical_actions: Dict[str, int]  # Actions performed and frequency
    peer_group: str
    activity_baseline: float  # Average daily events
    created_at: datetime
    updated_at: datetime


@dataclass
class RequestFlow:
    """Represents a request path through distributed systems"""
    request_id: str
    timestamp: datetime
    service_path: List[str]
    latencies: List[float]
    status_codes: List[int]
    payload_sizes: List[int]
    source: str
    success: bool


@dataclass
class BehavioralAnomaly:
    """Detected behavioral anomaly"""
    entity_id: str
    entity_type: str
    timestamp: datetime
    anomaly_type: str
    anomaly_score: float
    description: str
    evidence: Dict[str, Any]
    confidence: float


class UserBehaviorAnalyzer:
    """
    Analyzes user behavior patterns and detects anomalies.
    Implements UBA/UEBA techniques with profile-based detection.
    """
    
    def __init__(self, bedrock_client):
        """Initialize with AWS Bedrock client"""
        self.bedrock_client = bedrock_client
        self.user_profiles: Dict[str, UserProfile] = {}
        self.peer_groups: Dict[str, List[str]] = defaultdict(list)
    
    def build_user_profile(self, events: List[UserEvent], user_id: str) -> UserProfile:
        """
        Build a behavioral profile from historical user events.
        
        Args:
            events: Historical events for the user
            user_id: User identifier
            
        Returns:
            UserProfile with learned behavioral patterns
        """
        if not events:
            raise ValueError(f"No events provided for user {user_id}")
        
        # Extract temporal patterns
        hours = [e.timestamp.hour for e in events]
        typical_hours = [h for h, count in Counter(hours).items() 
                        if count > len(hours) * 0.05]  # Hours with >5% of activity
        
        # Extract resource access patterns
        resources = Counter(e.resource for e in events)
        
        # Extract action patterns
        actions = Counter(e.action for e in events)
        
        # Calculate activity baseline
        days = (max(e.timestamp for e in events) - 
                min(e.timestamp for e in events)).days + 1
        activity_baseline = len(events) / days if days > 0 else len(events)
        
        # Determine peer group (simplified - in production use role/dept)
        peer_group = self._assign_peer_group(events)
        
        profile = UserProfile(
            user_id=user_id,
            typical_hours=sorted(typical_hours),
            typical_resources=dict(resources),
            typical_actions=dict(actions),
            peer_group=peer_group,
            activity_baseline=activity_baseline,
            created_at=min(e.timestamp for e in events),
            updated_at=datetime.now()
        )
        
        self.user_profiles[user_id] = profile
        self.peer_groups[peer_group].append(user_id)
        
        return profile
    
    def _assign_peer_group(self, events: List[UserEvent]) -> str:
        """Assign user to a peer group based on behavior patterns"""
        # Simplified peer group assignment
        # In production, use role, department, access patterns
        resource_types = set(e.resource.split('/')[0] for e in events if '/' in e.resource)
        
        if 'admin' in resource_types:
            return 'administrators'
        elif 'api' in resource_types:
            return 'developers'
        elif 'database' in resource_types:
            return 'data_team'
        else:
            return 'general_users'
    
    def detect_user_anomalies(self, 
                             recent_events: List[UserEvent],
                             user_id: str,
                             window_hours: int = 24) -> List[BehavioralAnomaly]:
        """
        Detect behavioral anomalies for a user.
        
        Args:
            recent_events: Recent events to analyze
            user_id: User identifier
            window_hours: Time window for analysis
            
        Returns:
            List of detected anomalies
        """
        if user_id not in self.user_profiles:
            return []  # No profile yet - cold start
        
        profile = self.user_profiles[user_id]
        anomalies = []
        
        # Temporal anomaly detection
        temporal_anomaly = self._detect_temporal_anomaly(recent_events, profile)
        if temporal_anomaly:
            anomalies.append(temporal_anomaly)
        
        # Access pattern anomaly detection
        access_anomaly = self._detect_access_anomaly(recent_events, profile)
        if access_anomaly:
            anomalies.append(access_anomaly)
        
        # Volume anomaly detection
        volume_anomaly = self._detect_volume_anomaly(recent_events, profile, window_hours)
        if volume_anomaly:
            anomalies.append(volume_anomaly)
        
        # Sequence anomaly detection
        sequence_anomaly = self._detect_sequence_anomaly(recent_events, profile)
        if sequence_anomaly:
            anomalies.append(sequence_anomaly)
        
        # Peer group comparison
        peer_anomaly = self._detect_peer_deviation(recent_events, profile)
        if peer_anomaly:
            anomalies.append(peer_anomaly)
        
        return anomalies
    
    def _detect_temporal_anomaly(self, 
                                events: List[UserEvent],
                                profile: UserProfile) -> Optional[BehavioralAnomaly]:
        """Detect unusual activity times"""
        unusual_hours = [e.timestamp.hour for e in events 
                        if e.timestamp.hour not in profile.typical_hours]
        
        if len(unusual_hours) > len(events) * 0.3:  # >30% outside normal hours
            anomaly_score = len(unusual_hours) / len(events)
            
            return BehavioralAnomaly(
                entity_id=profile.user_id,
                entity_type='user',
                timestamp=datetime.now(),
                anomaly_type='temporal',
                anomaly_score=anomaly_score,
                description=f"User active during unusual hours",
                evidence={
                    'unusual_hours': unusual_hours,
                    'typical_hours': profile.typical_hours,
                    'percentage_unusual': anomaly_score * 100
                },
                confidence=0.85
            )
        
        return None
    
    def _detect_access_anomaly(self,
                              events: List[UserEvent],
                              profile: UserProfile) -> Optional[BehavioralAnomaly]:
        """Detect access to unusual resources"""
        accessed_resources = set(e.resource for e in events)
        typical_resources = set(profile.typical_resources.keys())
        
        new_resources = accessed_resources - typical_resources
        
        if new_resources and len(new_resources) > 3:  # Accessing multiple new resources
            anomaly_score = len(new_resources) / len(accessed_resources)
            
            return BehavioralAnomaly(
                entity_id=profile.user_id,
                entity_type='user',
                timestamp=datetime.now(),
                anomaly_type='access_pattern',
                anomaly_score=anomaly_score,
                description=f"User accessing {len(new_resources)} previously unused resources",
                evidence={
                    'new_resources': list(new_resources),
                    'typical_resource_count': len(typical_resources)
                },
                confidence=0.80
            )
        
        return None
    
    def _detect_volume_anomaly(self,
                              events: List[UserEvent],
                              profile: UserProfile,
                              window_hours: int) -> Optional[BehavioralAnomaly]:
        """Detect unusual activity volume"""
        events_per_day = (len(events) / window_hours) * 24
        
        # Use 3-sigma rule: beyond 3 standard deviations is anomalous
        threshold = profile.activity_baseline * 3
        
        if events_per_day > threshold:
            anomaly_score = min(events_per_day / threshold, 1.0)
            
            return BehavioralAnomaly(
                entity_id=profile.user_id,
                entity_type='user',
                timestamp=datetime.now(),
                anomaly_type='volume',
                anomaly_score=anomaly_score,
                description=f"Activity volume {anomaly_score:.1%} above normal",
                evidence={
                    'current_rate': events_per_day,
                    'baseline': profile.activity_baseline,
                    'threshold': threshold
                },
                confidence=0.90
            )
        
        return None
    
    def _detect_sequence_anomaly(self,
                                events: List[UserEvent],
                                profile: UserProfile) -> Optional[BehavioralAnomaly]:
        """Detect unusual action sequences using Markov chain approach"""
        if len(events) < 3:
            return None
        
        # Build action sequence
        action_sequence = [e.action for e in sorted(events, key=lambda x: x.timestamp)]
        
        # Check for rapid authentication changes (suspicious pattern)
        auth_actions = ['login', 'logout', 'password_reset', 'mfa_challenge']
        auth_sequence = [a for a in action_sequence if a in auth_actions]
        
        if len(auth_sequence) > 5 and len(auth_sequence) > len(action_sequence) * 0.5:
            return BehavioralAnomaly(
                entity_id=profile.user_id,
                entity_type='user',
                timestamp=datetime.now(),
                anomaly_type='sequence',
                anomaly_score=0.85,
                description="Unusual sequence of authentication actions",
                evidence={
                    'auth_sequence': auth_sequence,
                    'total_actions': len(action_sequence)
                },
                confidence=0.75
            )
        
        return None
    
    def _detect_peer_deviation(self,
                              events: List[UserEvent],
                              profile: UserProfile) -> Optional[BehavioralAnomaly]:
        """Compare user behavior to peer group"""
        peer_group = self.peer_groups.get(profile.peer_group, [])
        if len(peer_group) < 2:
            return None  # Not enough peers for comparison
        
        # Compare resource access
        user_resources = set(e.resource for e in events)
        
        # Get peer resources
        peer_resources = set()
        for peer_id in peer_group:
            if peer_id != profile.user_id and peer_id in self.user_profiles:
                peer_profile = self.user_profiles[peer_id]
                peer_resources.update(peer_profile.typical_resources.keys())
        
        # Resources this user accesses that no peer accesses
        unique_resources = user_resources - peer_resources
        
        if unique_resources and len(unique_resources) > 2:
            return BehavioralAnomaly(
                entity_id=profile.user_id,
                entity_type='user',
                timestamp=datetime.now(),
                anomaly_type='peer_deviation',
                anomaly_score=len(unique_resources) / len(user_resources),
                description=f"Accessing {len(unique_resources)} resources not used by peer group",
                evidence={
                    'unique_resources': list(unique_resources),
                    'peer_group': profile.peer_group,
                    'peer_count': len(peer_group)
                },
                confidence=0.70
            )
        
        return None
    
    def analyze_anomaly_with_llm(self, anomaly: BehavioralAnomaly) -> str:
        """
        Use AWS Bedrock Claude to provide context-aware analysis of anomaly.
        
        Args:
            anomaly: Detected behavioral anomaly
            
        Returns:
            Natural language analysis and recommended actions
        """
        prompt = f"""Analyze this behavioral security anomaly:

Entity: {anomaly.entity_id} ({anomaly.entity_type})
Anomaly Type: {anomaly.anomaly_type}
Score: {anomaly.anomaly_score:.2f}
Description: {anomaly.description}

Evidence:
{json.dumps(anomaly.evidence, indent=2)}

Provide:
1. Risk assessment (Low/Medium/High)
2. Possible explanations (both legitimate and malicious)
3. Recommended immediate actions
4. Additional data to investigate

Be specific and actionable."""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1000,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']


class RequestPatternAnalyzer:
    """
    Analyzes request flow patterns through distributed systems.
    Detects unusual service call sequences and timing anomalies.
    """
    
    def __init__(self, bedrock_client):
        """Initialize with AWS Bedrock client"""
        self.bedrock_client = bedrock_client
        self.normal_paths: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.latency_profiles: Dict[str, List[float]] = defaultdict(list)
    
    def learn_normal_paths(self, flows: List[RequestFlow]):
        """
        Learn normal request paths and timing from historical data.
        
        Args:
            flows: Historical request flows
        """
        for flow in flows:
            # Learn service call sequences
            for i in range(len(flow.service_path) - 1):
                current = flow.service_path[i]
                next_service = flow.service_path[i + 1]
                self.normal_paths[current][next_service] += 1
            
            # Learn latency distributions
            path_key = '->'.join(flow.service_path)
            total_latency = sum(flow.latencies)
            self.latency_profiles[path_key].append(total_latency)
    
    def detect_request_anomalies(self, flow: RequestFlow) -> List[BehavioralAnomaly]:
        """
        Detect anomalies in request flow patterns.
        
        Args:
            flow: Request flow to analyze
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for unusual service sequences
        sequence_anomaly = self._detect_path_anomaly(flow)
        if sequence_anomaly:
            anomalies.append(sequence_anomaly)
        
        # Check for timing anomalies
        timing_anomaly = self._detect_timing_anomaly(flow)
        if timing_anomaly:
            anomalies.append(timing_anomaly)
        
        # Check for payload anomalies
        payload_anomaly = self._detect_payload_anomaly(flow)
        if payload_anomaly:
            anomalies.append(payload_anomaly)
        
        return anomalies
    
    def _detect_path_anomaly(self, flow: RequestFlow) -> Optional[BehavioralAnomaly]:
        """Detect unusual service call sequences"""
        unusual_transitions = []
        
        for i in range(len(flow.service_path) - 1):
            current = flow.service_path[i]
            next_service = flow.service_path[i + 1]
            
            # Check if this transition has been seen before
            if current in self.normal_paths:
                if next_service not in self.normal_paths[current]:
                    unusual_transitions.append(f"{current}->{next_service}")
        
        if unusual_transitions:
            return BehavioralAnomaly(
                entity_id=flow.request_id,
                entity_type='request',
                timestamp=flow.timestamp,
                anomaly_type='path',
                anomaly_score=len(unusual_transitions) / (len(flow.service_path) - 1),
                description=f"Request took {len(unusual_transitions)} unusual service transitions",
                evidence={
                    'unusual_transitions': unusual_transitions,
                    'full_path': flow.service_path,
                    'source': flow.source
                },
                confidence=0.80
            )
        
        return None
    
    def _detect_timing_anomaly(self, flow: RequestFlow) -> Optional[BehavioralAnomaly]:
        """Detect unusual request timing"""
        path_key = '->'.join(flow.service_path)
        
        if path_key not in self.latency_profiles or len(self.latency_profiles[path_key]) < 10:
            return None  # Not enough data
        
        historical_latencies = self.latency_profiles[path_key]
        mean_latency = np.mean(historical_latencies)
        std_latency = np.std(historical_latencies)
        
        current_latency = sum(flow.latencies)
        
        # Z-score approach
        if std_latency > 0:
            z_score = abs((current_latency - mean_latency) / std_latency)
            
            if z_score > 3:  # Beyond 3 standard deviations
                return BehavioralAnomaly(
                    entity_id=flow.request_id,
                    entity_type='request',
                    timestamp=flow.timestamp,
                    anomaly_type='timing',
                    anomaly_score=min(z_score / 5, 1.0),
                    description=f"Request latency {z_score:.1f} std devs from normal",
                    evidence={
                        'current_latency_ms': current_latency,
                        'mean_latency_ms': mean_latency,
                        'std_latency_ms': std_latency,
                        'z_score': z_score,
                        'service_latencies': dict(zip(flow.service_path, flow.latencies))
                    },
                    confidence=0.85
                )
        
        return None
    
    def _detect_payload_anomaly(self, flow: RequestFlow) -> Optional[BehavioralAnomaly]:
        """Detect unusual payload sizes"""
        if not flow.payload_sizes:
            return None
        
        total_payload = sum(flow.payload_sizes)
        
        # Simple threshold-based detection (in production, use learned distributions)
        # Flag payloads over 10MB as potentially anomalous
        if total_payload > 10 * 1024 * 1024:
            return BehavioralAnomaly(
                entity_id=flow.request_id,
                entity_type='request',
                timestamp=flow.timestamp,
                anomaly_type='payload',
                anomaly_score=min(total_payload / (50 * 1024 * 1024), 1.0),
                description=f"Unusually large payload: {total_payload / (1024*1024):.1f} MB",
                evidence={
                    'total_bytes': total_payload,
                    'service_payloads': dict(zip(flow.service_path, flow.payload_sizes))
                },
                confidence=0.75
            )
        
        return None
    
    def generate_path_report(self, flows: List[RequestFlow]) -> str:
        """
        Generate a summary report of request path patterns using AWS Bedrock.
        
        Args:
            flows: Recent request flows
            
        Returns:
            Natural language report
        """
        # Collect path statistics
        path_stats = defaultdict(int)
        error_paths = []
        
        for flow in flows:
            path = '->'.join(flow.service_path)
            path_stats[path] += 1
            
            if not flow.success:
                error_paths.append({
                    'path': path,
                    'status_codes': flow.status_codes,
                    'latencies': flow.latencies
                })
        
        # Get top paths
        top_paths = sorted(path_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        
        prompt = f"""Analyze these request path patterns:

Total Requests: {len(flows)}
Unique Paths: {len(path_stats)}

Top Request Paths:
{chr(10).join(f"{i+1}. {path} ({count} requests)" for i, (path, count) in enumerate(top_paths))}

Failed Request Paths ({len(error_paths)} failures):
{json.dumps(error_paths[:5], indent=2)}

Provide:
1. Overall system health assessment
2. Unusual patterns or concerns
3. Services appearing in failure paths
4. Recommendations for investigation"""

        response = self.bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1500,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']


class GraphAnomalyDetector:
    """
    Graph-based anomaly detection for entity relationships and communication patterns.
    Uses graph structure to detect unusual connections and patterns.
    """
    
    def __init__(self, bedrock_client):
        """Initialize with AWS Bedrock client"""
        self.bedrock_client = bedrock_client
        self.graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.node_degrees: Dict[str, int] = defaultdict(int)
    
    def build_communication_graph(self, flows: List[RequestFlow]):
        """
        Build a communication graph from request flows.
        
        Args:
            flows: Request flows showing service-to-service communication
        """
        for flow in flows:
            for i in range(len(flow.service_path) - 1):
                source = flow.service_path[i]
                target = flow.service_path[i + 1]
                
                self.graph[source][target] += 1
                self.node_degrees[source] += 1
                self.node_degrees[target] += 1
    
    def detect_graph_anomalies(self, recent_flows: List[RequestFlow]) -> List[BehavioralAnomaly]:
        """
        Detect graph-based anomalies in recent communications.
        
        Args:
            recent_flows: Recent request flows
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Build temporary graph from recent data
        recent_graph = defaultdict(lambda: defaultdict(int))
        for flow in recent_flows:
            for i in range(len(flow.service_path) - 1):
                source = flow.service_path[i]
                target = flow.service_path[i + 1]
                recent_graph[source][target] += 1
        
        # Detect new edges
        new_edge_anomaly = self._detect_new_edges(recent_graph)
        if new_edge_anomaly:
            anomalies.append(new_edge_anomaly)
        
        # Detect unusual node connectivity
        connectivity_anomaly = self._detect_connectivity_change(recent_graph)
        if connectivity_anomaly:
            anomalies.append(connectivity_anomaly)
        
        return anomalies
    
    def _detect_new_edges(self, recent_graph: Dict[str, Dict[str, int]]) -> Optional[BehavioralAnomaly]:
        """Detect new communication edges not seen in historical graph"""
        new_edges = []
        
        for source, targets in recent_graph.items():
            for target in targets:
                # Check if this edge exists in historical graph
                if source not in self.graph or target not in self.graph[source]:
                    new_edges.append(f"{source}->{target}")
        
        if new_edges:
            return BehavioralAnomaly(
                entity_id='system_graph',
                entity_type='graph',
                timestamp=datetime.now(),
                anomaly_type='new_edges',
                anomaly_score=min(len(new_edges) / 10, 1.0),
                description=f"Detected {len(new_edges)} new service communication patterns",
                evidence={
                    'new_edges': new_edges,
                    'total_edges': sum(len(targets) for targets in recent_graph.values())
                },
                confidence=0.75
            )
        
        return None
    
    def _detect_connectivity_change(self, recent_graph: Dict[str, Dict[str, int]]) -> Optional[BehavioralAnomaly]:
        """Detect significant changes in node connectivity"""
        connectivity_changes = []
        
        for node in recent_graph:
            if node in self.node_degrees:
                # Count connections in recent graph
                recent_degree = len(recent_graph[node])
                historical_degree = len(self.graph.get(node, {}))
                
                # Flag if connectivity changed significantly
                if historical_degree > 0:
                    change_ratio = abs(recent_degree - historical_degree) / historical_degree
                    
                    if change_ratio > 0.5:  # 50% change in connectivity
                        connectivity_changes.append({
                            'node': node,
                            'historical_connections': historical_degree,
                            'recent_connections': recent_degree,
                            'change_ratio': change_ratio
                        })
        
        if connectivity_changes:
            return BehavioralAnomaly(
                entity_id='system_graph',
                entity_type='graph',
                timestamp=datetime.now(),
                anomaly_type='connectivity_change',
                anomaly_score=min(len(connectivity_changes) / 5, 1.0),
                description=f"{len(connectivity_changes)} services show significant connectivity changes",
                evidence={
                    'affected_services': connectivity_changes
                },
                confidence=0.70
            )
        
        return None
    
    def visualize_anomalous_patterns(self, anomalies: List[BehavioralAnomaly]) -> str:
        """
        Generate natural language visualization of graph anomalies using AWS Bedrock.
        
        Args:
            anomalies: Detected graph anomalies
            
        Returns:
            Natural language description of graph changes
        """
        if not anomalies:
            return "No significant graph anomalies detected."
        
        anomaly_summaries = []
        for anomaly in anomalies:
            anomaly_summaries.append({
                'type': anomaly.anomaly_type,
                'score': anomaly.anomaly_score,
                'description': anomaly.description,
                'evidence': anomaly.evidence
            })
        
        prompt = f"""Analyze these service communication graph anomalies:

Detected Anomalies:
{json.dumps(anomaly_summaries, indent=2)}

Provide:
1. Summary of what changed in the service communication pattern
2. Potential causes (deployments, failures, attacks)
3. Services that may be affected or at risk
4. Recommended investigation steps

Focus on explaining the graph structure changes in clear terms."""

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


class BehavioralDetectionPipeline:
    """
    Production pipeline orchestrating multiple behavioral detection approaches.
    Combines user, request, and graph-based detection with intelligent alerting.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize the behavioral detection pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.user_analyzer = UserBehaviorAnalyzer(self.bedrock_client)
        self.request_analyzer = RequestPatternAnalyzer(self.bedrock_client)
        self.graph_detector = GraphAnomalyDetector(self.bedrock_client)
        
        self.anomaly_history: List[BehavioralAnomaly] = []
    
    def train_baseline(self,
                      historical_events: List[UserEvent],
                      historical_flows: List[RequestFlow]):
        """
        Train baseline models from historical data.
        
        Args:
            historical_events: Historical user events
            historical_flows: Historical request flows
        """
        print("Training behavioral baselines...")
        
        # Build user profiles
        users = set(e.user_id for e in historical_events)
        for user_id in users:
            user_events = [e for e in historical_events if e.user_id == user_id]
            if len(user_events) >= 10:  # Minimum events for profile
                self.user_analyzer.build_user_profile(user_events, user_id)
        
        print(f"Built profiles for {len(self.user_analyzer.user_profiles)} users")
        
        # Learn normal request paths
        self.request_analyzer.learn_normal_paths(historical_flows)
        print(f"Learned {len(self.request_analyzer.normal_paths)} service transitions")
        
        # Build communication graph
        self.graph_detector.build_communication_graph(historical_flows)
        print(f"Built graph with {len(self.graph_detector.graph)} nodes")
    
    def analyze_current_behavior(self,
                                recent_events: List[UserEvent],
                                recent_flows: List[RequestFlow]) -> Dict[str, Any]:
        """
        Analyze recent behavior across all detection methods.
        
        Args:
            recent_events: Recent user events
            recent_flows: Recent request flows
            
        Returns:
            Comprehensive analysis results
        """
        all_anomalies = []
        
        # User behavior analysis
        users = set(e.user_id for e in recent_events)
        for user_id in users:
            user_events = [e for e in recent_events if e.user_id == user_id]
            anomalies = self.user_analyzer.detect_user_anomalies(user_events, user_id)
            all_anomalies.extend(anomalies)
        
        # Request pattern analysis
        for flow in recent_flows:
            anomalies = self.request_analyzer.detect_request_anomalies(flow)
            all_anomalies.extend(anomalies)
        
        # Graph analysis
        graph_anomalies = self.graph_detector.detect_graph_anomalies(recent_flows)
        all_anomalies.extend(graph_anomalies)
        
        # Store in history
        self.anomaly_history.extend(all_anomalies)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(all_anomalies, recent_events, recent_flows)
        
        return {
            'anomalies': all_anomalies,
            'anomaly_count': len(all_anomalies),
            'high_severity': [a for a in all_anomalies if a.anomaly_score > 0.8],
            'by_type': self._group_by_type(all_anomalies),
            'report': report
        }
    
    def _group_by_type(self, anomalies: List[BehavioralAnomaly]) -> Dict[str, int]:
        """Group anomalies by type"""
        type_counts = defaultdict(int)
        for anomaly in anomalies:
            type_counts[anomaly.anomaly_type] += 1
        return dict(type_counts)
    
    def _generate_comprehensive_report(self,
                                      anomalies: List[BehavioralAnomaly],
                                      events: List[UserEvent],
                                      flows: List[RequestFlow]) -> str:
        """
        Generate comprehensive behavioral analysis report using AWS Bedrock.
        
        Args:
            anomalies: All detected anomalies
            events: Recent user events
            flows: Recent request flows
            
        Returns:
            Natural language comprehensive report
        """
        if not anomalies:
            return "No behavioral anomalies detected in current analysis window."
        
        # Prepare summary statistics
        anomaly_summary = []
        for anomaly in sorted(anomalies, key=lambda x: x.anomaly_score, reverse=True)[:10]:
            anomaly_summary.append({
                'entity': anomaly.entity_id,
                'type': anomaly.anomaly_type,
                'score': round(anomaly.anomaly_score, 3),
                'description': anomaly.description
            })
        
        user_count = len(set(e.user_id for e in events))
        flow_count = len(flows)
        
        prompt = f"""Analyze this comprehensive behavioral security report:

Time Window: Last 24 hours
Users Analyzed: {user_count}
Requests Analyzed: {flow_count}
Anomalies Detected: {len(anomalies)}

Top Anomalies by Severity:
{json.dumps(anomaly_summary, indent=2)}

Anomalies by Type:
{json.dumps(self._group_by_type(anomalies), indent=2)}

Provide:
1. Executive summary of behavioral security posture
2. Most concerning findings requiring immediate attention
3. Patterns across multiple anomalies that suggest coordinated issues
4. Recommended priority actions for security and ops teams
5. Areas for follow-up investigation

Be specific, actionable, and prioritize by business impact."""

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


# Example usage and realistic data generators
def generate_sample_user_events(count: int = 100) -> List[UserEvent]:
    """Generate realistic sample user events for testing"""
    users = ['alice', 'bob', 'charlie', 'david', 'eve']
    actions = ['login', 'view_dashboard', 'run_query', 'export_data', 'logout',
               'update_config', 'create_report', 'delete_resource']
    resources = ['dashboard/sales', 'database/customers', 'api/analytics',
                'admin/users', 'reports/financial', 'config/system']
    
    events = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(count):
        # Simulate normal working hours (9-5) with occasional off-hours
        hour_offset = np.random.choice([9, 10, 11, 12, 13, 14, 15, 16, 17], p=[0.1, 0.12, 0.12, 0.1, 0.12, 0.12, 0.12, 0.1, 0.1])
        if np.random.random() < 0.05:  # 5% outside normal hours
            hour_offset = np.random.randint(0, 24)
        
        timestamp = base_time + timedelta(
            hours=hour_offset,
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        
        user_id = np.random.choice(users)
        
        # Alice occasionally has suspicious behavior
        if user_id == 'alice' and np.random.random() < 0.1:
            action = 'admin/privileged'
            resource = 'database/sensitive'
        else:
            action = np.random.choice(actions)
            resource = np.random.choice(resources)
        
        events.append(UserEvent(
            user_id=user_id,
            timestamp=timestamp,
            action=action,
            resource=resource,
            source_ip=f"192.168.1.{np.random.randint(1, 255)}",
            success=np.random.random() > 0.05,  # 95% success rate
            metadata={'session_id': f"sess_{i}"}
        ))
    
    return sorted(events, key=lambda x: x.timestamp)


def generate_sample_request_flows(count: int = 50) -> List[RequestFlow]:
    """Generate realistic sample request flows for testing"""
    services = ['frontend', 'api-gateway', 'auth-service', 'user-service',
                'database', 'cache', 'payment-service', 'notification-service']
    
    common_paths = [
        ['frontend', 'api-gateway', 'user-service', 'database'],
        ['frontend', 'api-gateway', 'auth-service', 'database'],
        ['api-gateway', 'payment-service', 'database', 'notification-service'],
        ['frontend', 'api-gateway', 'cache', 'user-service'],
    ]
    
    flows = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(count):
        # Most requests follow common paths
        if np.random.random() < 0.9:
            path = np.random.choice(common_paths).copy()
        else:
            # Anomalous path
            path = np.random.choice(services, size=np.random.randint(3, 6), replace=False).tolist()
        
        # Generate latencies (normally 10-100ms per service)
        latencies = [max(5, np.random.normal(50, 20)) for _ in path]
        
        # Occasionally add high latency (anomaly)
        if np.random.random() < 0.05:
            latencies[-1] *= 10  # 10x latency spike
        
        flows.append(RequestFlow(
            request_id=f"req_{i:04d}",
            timestamp=base_time + timedelta(minutes=i),
            service_path=path,
            latencies=latencies,
            status_codes=[200] * len(path) if np.random.random() > 0.1 else [200] * (len(path)-1) + [500],
            payload_sizes=[np.random.randint(1024, 10240) for _ in path],
            source=f"client_{np.random.randint(1, 20)}",
            success=np.random.random() > 0.1
        ))
    
    return flows


def main():
    """
    Demonstrate behavioral anomaly detection pipeline with realistic data.
    """
    print("=" * 80)
    print("Chapter 9: Behavioral Anomaly Detection")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing behavioral detection pipeline...")
    pipeline = BehavioralDetectionPipeline(aws_region='us-east-1')
    print()
    
    # Generate historical data for training
    print("Generating historical data for baseline training...")
    historical_events = generate_sample_user_events(count=500)
    historical_flows = generate_sample_request_flows(count=200)
    print(f"Generated {len(historical_events)} historical user events")
    print(f"Generated {len(historical_flows)} historical request flows")
    print()
    
    # Train baselines
    print("Training behavioral baselines...")
    pipeline.train_baseline(historical_events, historical_flows)
    print()
    
    # Generate recent data with some anomalies
    print("Generating recent data with anomalies...")
    recent_events = generate_sample_user_events(count=100)
    recent_flows = generate_sample_request_flows(count=50)
    
    # Add some explicit anomalies
    # Anomaly 1: User active at unusual hour
    recent_events.append(UserEvent(
        user_id='alice',
        timestamp=datetime.now().replace(hour=3, minute=0),  # 3 AM
        action='export_data',
        resource='database/sensitive',
        source_ip='192.168.1.99',
        success=True,
        metadata={'suspicious': True}
    ))
    
    # Anomaly 2: Unusual service path
    recent_flows.append(RequestFlow(
        request_id='req_anomaly_1',
        timestamp=datetime.now(),
        service_path=['frontend', 'secret-service', 'admin-db', 'external-api'],  # Unusual path
        latencies=[50, 75, 100, 200],
        status_codes=[200, 200, 200, 200],
        payload_sizes=[1024, 2048, 4096, 8192],
        source='unknown_client',
        success=True
    ))
    
    print(f"Generated {len(recent_events)} recent user events")
    print(f"Generated {len(recent_flows)} recent request flows")
    print()
    
    # Analyze current behavior
    print("Analyzing current behavioral patterns...")
    print("-" * 80)
    results = pipeline.analyze_current_behavior(recent_events, recent_flows)
    print()
    
    # Display results
    print(f"Total Anomalies Detected: {results['anomaly_count']}")
    print(f"High Severity Anomalies: {len(results['high_severity'])}")
    print()
    
    print("Anomalies by Type:")
    for anomaly_type, count in results['by_type'].items():
        print(f"  {anomaly_type}: {count}")
    print()
    
    # Show top anomalies
    if results['anomalies']:
        print("Top 5 Anomalies:")
        print("-" * 80)
        top_anomalies = sorted(results['anomalies'], 
                              key=lambda x: x.anomaly_score, 
                              reverse=True)[:5]
        
        for i, anomaly in enumerate(top_anomalies, 1):
            print(f"\n{i}. {anomaly.entity_type.upper()}: {anomaly.entity_id}")
            print(f"   Type: {anomaly.anomaly_type}")
            print(f"   Score: {anomaly.anomaly_score:.3f}")
            print(f"   Description: {anomaly.description}")
            print(f"   Confidence: {anomaly.confidence:.2%}")
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BEHAVIORAL ANALYSIS REPORT")
        print("=" * 80)
        print()
        print(results['report'])
        print()
        
        # Get detailed analysis for the top anomaly
        if results['high_severity']:
            print("\n" + "=" * 80)
            print("DETAILED ANALYSIS: Top Severity Anomaly")
            print("=" * 80)
            print()
            top_severity = results['high_severity'][0]
            detailed_analysis = pipeline.user_analyzer.analyze_anomaly_with_llm(top_severity)
            print(detailed_analysis)
    else:
        print("No anomalies detected - all behavior within expected patterns")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
