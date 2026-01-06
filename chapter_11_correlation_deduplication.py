"""
Chapter 11: Correlation and Deduplication
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready alert correlation and deduplication including:
- Exact, fuzzy, and temporal deduplication
- Topology-based correlation
- Temporal pattern correlation
- Semantic correlation using embeddings
- Graph-based correlation
- Incident aggregation with AI summarization
- Real-time correlation pipeline with AWS Bedrock

Author: AI-Powered Observability
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, field
import numpy as np
from scipy import stats
import hashlib
from difflib import SequenceMatcher
import networkx as nx


@dataclass
class Alert:
    """Represents an individual alert"""
    alert_id: str
    timestamp: datetime
    source: str
    severity: str  # critical, warning, info
    title: str
    message: str
    affected_entity: str
    tags: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    embedding: Optional[List[float]] = None


@dataclass
class Incident:
    """Represents a correlated group of alerts"""
    incident_id: str
    created_at: datetime
    updated_at: datetime
    alerts: List[Alert]
    primary_alert: Alert
    severity: str
    title: str
    summary: str
    affected_entities: Set[str]
    status: str  # active, investigating, resolved
    root_cause: Optional[str] = None
    correlation_confidence: float = 0.0


class AlertDeduplicator:
    """
    Implements exact, fuzzy, and temporal deduplication of alerts.
    """
    
    def __init__(self):
        """Initialize deduplicator"""
        self.active_alerts: Dict[str, Alert] = {}
        self.dedup_counts: Dict[str, int] = defaultdict(int)
    
    def exact_deduplicate(self, alert: Alert) -> Tuple[bool, Optional[str]]:
        """
        Exact deduplication based on key fields.
        
        Args:
            alert: Alert to check for duplication
            
        Returns:
            Tuple of (is_duplicate, existing_alert_id)
        """
        # Create deduplication key from stable fields
        dedup_key = self._create_dedup_key(alert)
        
        if dedup_key in self.active_alerts:
            existing_id = self.active_alerts[dedup_key].alert_id
            self.dedup_counts[dedup_key] += 1
            return True, existing_id
        
        self.active_alerts[dedup_key] = alert
        self.dedup_counts[dedup_key] = 1
        return False, None
    
    def _create_dedup_key(self, alert: Alert) -> str:
        """Create unique key for exact deduplication"""
        key_parts = [
            alert.source,
            alert.title,
            alert.affected_entity,
            alert.severity
        ]
        return hashlib.md5('::'.join(key_parts).encode()).hexdigest()
    
    def fuzzy_deduplicate(self, 
                         alert: Alert,
                         similarity_threshold: float = 0.85) -> Tuple[bool, Optional[str], float]:
        """
        Fuzzy deduplication using text similarity.
        
        Args:
            alert: Alert to check
            similarity_threshold: Minimum similarity to consider duplicate
            
        Returns:
            Tuple of (is_duplicate, existing_alert_id, similarity_score)
        """
        best_match_id = None
        best_similarity = 0.0
        
        # Compare against active alerts
        for existing_alert in self.active_alerts.values():
            # Skip if different source or entity
            if (existing_alert.source != alert.source or 
                existing_alert.affected_entity != alert.affected_entity):
                continue
            
            # Calculate text similarity
            similarity = self._calculate_similarity(
                alert.message,
                existing_alert.message
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = existing_alert.alert_id
        
        is_duplicate = best_similarity >= similarity_threshold
        return is_duplicate, best_match_id, best_similarity
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def temporal_deduplicate(self,
                           alert: Alert,
                           flap_window_minutes: int = 15) -> Tuple[bool, Optional[str]]:
        """
        Temporal deduplication to handle flapping alerts.
        
        Args:
            alert: Alert to check
            flap_window_minutes: Time window for considering flaps
            
        Returns:
            Tuple of (is_flapping, original_alert_id)
        """
        dedup_key = self._create_dedup_key(alert)
        
        # Check if we've seen this alert recently
        if dedup_key in self.active_alerts:
            existing_alert = self.active_alerts[dedup_key]
            
            # If it resolved and re-fired within the window, it's flapping
            if existing_alert.resolved:
                time_since_resolution = (alert.timestamp - 
                                       existing_alert.resolved_at).total_seconds() / 60
                
                if time_since_resolution <= flap_window_minutes:
                    return True, existing_alert.alert_id
        
        return False, None
    
    def resolve_alert(self, alert_id: str, resolved_at: datetime):
        """Mark an alert as resolved"""
        for alert in self.active_alerts.values():
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = resolved_at
                break
    
    def get_dedup_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        total_alerts = sum(self.dedup_counts.values())
        unique_alerts = len(self.dedup_counts)
        
        return {
            'total_alerts_seen': total_alerts,
            'unique_alerts': unique_alerts,
            'duplicate_count': total_alerts - unique_alerts,
            'dedup_rate': (total_alerts - unique_alerts) / total_alerts if total_alerts > 0 else 0,
            'top_repeating_alerts': sorted(
                [(k, v) for k, v in self.dedup_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


class TopologyCorrelator:
    """
    Correlates alerts based on service dependency topology.
    """
    
    def __init__(self):
        """Initialize topology correlator"""
        self.topology_graph = nx.DiGraph()
    
    def add_dependency(self, service: str, depends_on: str):
        """
        Add a service dependency to the topology.
        
        Args:
            service: Service name
            depends_on: Service that this service depends on
        """
        self.topology_graph.add_edge(service, depends_on)
    
    def build_topology_from_alerts(self, alerts: List[Alert]):
        """Learn topology from alert co-occurrence patterns"""
        # Build co-occurrence matrix
        entity_pairs = defaultdict(int)
        
        for i, alert1 in enumerate(alerts):
            for alert2 in alerts[i+1:]:
                # If alerts are close in time, they might be related
                time_diff = abs((alert1.timestamp - alert2.timestamp).total_seconds())
                if time_diff <= 300:  # 5 minutes
                    pair = tuple(sorted([alert1.affected_entity, alert2.affected_entity]))
                    entity_pairs[pair] += 1
        
        # Add edges for frequently co-occurring entities
        for (entity1, entity2), count in entity_pairs.items():
            if count >= 3:  # Threshold for adding dependency
                self.topology_graph.add_edge(entity1, entity2)
    
    def correlate_by_topology(self, alerts: List[Alert]) -> List[List[Alert]]:
        """
        Group alerts based on topology relationships.
        
        Args:
            alerts: List of alerts to correlate
            
        Returns:
            List of alert groups
        """
        # Map alerts to their entities
        entity_alerts = defaultdict(list)
        for alert in alerts:
            entity_alerts[alert.affected_entity].append(alert)
        
        # Find connected components in the topology
        alert_groups = []
        visited_entities = set()
        
        for entity in entity_alerts.keys():
            if entity not in visited_entities:
                # Find all entities connected to this one
                if entity in self.topology_graph:
                    connected = nx.node_connected_component(
                        self.topology_graph.to_undirected(),
                        entity
                    )
                else:
                    connected = {entity}
                
                # Collect all alerts from connected entities
                group = []
                for connected_entity in connected:
                    if connected_entity in entity_alerts:
                        group.extend(entity_alerts[connected_entity])
                        visited_entities.add(connected_entity)
                
                if group:
                    alert_groups.append(group)
        
        return alert_groups
    
    def identify_root_cause_alert(self, alert_group: List[Alert]) -> Optional[Alert]:
        """
        Identify the root cause alert in a correlated group.
        Uses topology to find the alert closest to the dependency root.
        
        Args:
            alert_group: Group of correlated alerts
            
        Returns:
            Alert most likely to be the root cause
        """
        if not alert_group:
            return None
        
        # Score alerts by their position in the dependency graph
        alert_scores = []
        
        for alert in alert_group:
            entity = alert.affected_entity
            
            if entity in self.topology_graph:
                # Count how many services depend on this one (outgoing edges)
                dependents = len(list(self.topology_graph.successors(entity)))
                
                # Count dependencies of this service (incoming edges)
                dependencies = len(list(self.topology_graph.predecessors(entity)))
                
                # Root causes tend to have more dependents and fewer dependencies
                score = dependents - dependencies * 0.5
            else:
                score = 0
            
            alert_scores.append((alert, score))
        
        # Return alert with highest score
        return max(alert_scores, key=lambda x: x[1])[0]


class TemporalCorrelator:
    """
    Correlates alerts based on temporal patterns.
    """
    
    def __init__(self, base_window_minutes: int = 5):
        """
        Initialize temporal correlator.
        
        Args:
            base_window_minutes: Base time window for correlation
        """
        self.base_window = timedelta(minutes=base_window_minutes)
        self.learned_windows: Dict[str, timedelta] = {}
    
    def correlate_by_time(self, alerts: List[Alert]) -> List[List[Alert]]:
        """
        Group alerts that occur close together in time.
        
        Args:
            alerts: List of alerts to correlate
            
        Returns:
            List of temporally-related alert groups
        """
        if not alerts:
            return []
        
        # Sort alerts by timestamp
        sorted_alerts = sorted(alerts, key=lambda a: a.timestamp)
        
        groups = []
        current_group = [sorted_alerts[0]]
        
        for alert in sorted_alerts[1:]:
            # Get appropriate window for this alert type
            window = self._get_correlation_window(alert)
            
            # Check if this alert is within the window of the last alert in current group
            time_diff = alert.timestamp - current_group[-1].timestamp
            
            if time_diff <= window:
                current_group.append(alert)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [alert]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _get_correlation_window(self, alert: Alert) -> timedelta:
        """Get appropriate correlation window for an alert type"""
        alert_type = f"{alert.source}::{alert.title}"
        
        if alert_type in self.learned_windows:
            return self.learned_windows[alert_type]
        
        return self.base_window
    
    def learn_cascade_timing(self, historical_incidents: List[List[Alert]]):
        """
        Learn typical cascade timing patterns from historical incidents.
        
        Args:
            historical_incidents: List of historical alert groups
        """
        # Analyze timing patterns in historical incidents
        alert_type_timings = defaultdict(list)
        
        for incident_alerts in historical_incidents:
            if len(incident_alerts) < 2:
                continue
            
            sorted_alerts = sorted(incident_alerts, key=lambda a: a.timestamp)
            
            for i in range(len(sorted_alerts) - 1):
                alert_type = f"{sorted_alerts[i].source}::{sorted_alerts[i].title}"
                time_to_next = (sorted_alerts[i + 1].timestamp - 
                              sorted_alerts[i].timestamp).total_seconds()
                alert_type_timings[alert_type].append(time_to_next)
        
        # Compute learned windows (mean + 2 std devs)
        for alert_type, timings in alert_type_timings.items():
            if timings:
                mean_timing = np.mean(timings)
                std_timing = np.std(timings)
                window_seconds = mean_timing + 2 * std_timing
                self.learned_windows[alert_type] = timedelta(seconds=window_seconds)


class SemanticCorrelator:
    """
    Correlates alerts based on semantic similarity using embeddings.
    """
    
    def __init__(self, bedrock_client):
        """
        Initialize semantic correlator.
        
        Args:
            bedrock_client: AWS Bedrock client for embeddings
        """
        self.bedrock_client = bedrock_client
        self.alert_embeddings: Dict[str, List[float]] = {}
    
    def generate_alert_embedding(self, alert: Alert) -> List[float]:
        """
        Generate embedding for an alert.
        
        Args:
            alert: Alert to embed
            
        Returns:
            Embedding vector
        """
        # Create text representation of alert
        alert_text = f"{alert.title}. {alert.message}. Source: {alert.source}. Entity: {alert.affected_entity}"
        
        # Generate embedding using AWS Bedrock Titan
        response = self.bedrock_client.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=json.dumps({
                'inputText': alert_text
            })
        )
        
        result = json.loads(response['body'].read())
        embedding = result['embedding']
        
        # Cache the embedding
        self.alert_embeddings[alert.alert_id] = embedding
        alert.embedding = embedding
        
        return embedding
    
    def correlate_by_semantics(self,
                              alerts: List[Alert],
                              similarity_threshold: float = 0.75) -> List[List[Alert]]:
        """
        Group alerts based on semantic similarity.
        
        Args:
            alerts: List of alerts to correlate
            similarity_threshold: Minimum cosine similarity for correlation
            
        Returns:
            List of semantically-related alert groups
        """
        # Generate embeddings for all alerts
        for alert in alerts:
            if alert.embedding is None:
                self.generate_alert_embedding(alert)
        
        # Build similarity matrix
        n = len(alerts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._cosine_similarity(
                    alerts[i].embedding,
                    alerts[j].embedding
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # Group alerts by similarity
        groups = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Find all alerts similar to this one
            group = [alerts[i]]
            assigned.add(i)
            
            for j in range(i + 1, n):
                if j not in assigned and similarity_matrix[i][j] >= similarity_threshold:
                    group.append(alerts[j])
                    assigned.add(j)
            
            groups.append(group)
        
        return groups
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_arr = np.array(vec1)
        vec2_arr = np.array(vec2)
        
        dot_product = np.dot(vec1_arr, vec2_arr)
        norm1 = np.linalg.norm(vec1_arr)
        norm2 = np.linalg.norm(vec2_arr)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class GraphCorrelator:
    """
    Advanced correlation using graph-based approaches.
    """
    
    def __init__(self):
        """Initialize graph correlator"""
        self.correlation_graph = nx.Graph()
    
    def build_correlation_graph(self, alerts: List[Alert], correlators: Dict[str, Any]):
        """
        Build a graph representing alert relationships.
        
        Args:
            alerts: List of alerts
            correlators: Dictionary of correlators (temporal, topology, semantic)
        """
        self.correlation_graph.clear()
        
        # Add all alerts as nodes
        for alert in alerts:
            self.correlation_graph.add_node(alert.alert_id, alert=alert)
        
        # Add edges based on different correlation signals
        for i, alert1 in enumerate(alerts):
            for alert2 in alerts[i+1:]:
                edge_weight = 0.0
                edge_reasons = []
                
                # Temporal correlation
                if 'temporal' in correlators:
                    time_diff = abs((alert1.timestamp - alert2.timestamp).total_seconds())
                    if time_diff <= 300:  # 5 minutes
                        weight = max(0, 1 - time_diff / 300)
                        edge_weight += weight * 0.3
                        edge_reasons.append(f"temporal({weight:.2f})")
                
                # Topology correlation
                if 'topology' in correlators:
                    topology = correlators['topology']
                    if (alert1.affected_entity in topology.topology_graph and
                        alert2.affected_entity in topology.topology_graph):
                        try:
                            path_length = nx.shortest_path_length(
                                topology.topology_graph,
                                alert1.affected_entity,
                                alert2.affected_entity
                            )
                            if path_length <= 2:
                                weight = 1 - path_length / 3
                                edge_weight += weight * 0.4
                                edge_reasons.append(f"topology({weight:.2f})")
                        except nx.NetworkXNoPath:
                            pass
                
                # Semantic correlation
                if 'semantic' in correlators:
                    if alert1.embedding and alert2.embedding:
                        semantic_sim = correlators['semantic']._cosine_similarity(
                            alert1.embedding,
                            alert2.embedding
                        )
                        if semantic_sim > 0.7:
                            edge_weight += semantic_sim * 0.3
                            edge_reasons.append(f"semantic({semantic_sim:.2f})")
                
                # Add edge if there's any correlation signal
                if edge_weight > 0.3:  # Minimum threshold
                    self.correlation_graph.add_edge(
                        alert1.alert_id,
                        alert2.alert_id,
                        weight=edge_weight,
                        reasons=edge_reasons
                    )
    
    def detect_incident_groups(self, min_weight: float = 0.5) -> List[List[Alert]]:
        """
        Detect incident groups using community detection.
        
        Args:
            min_weight: Minimum edge weight to consider
            
        Returns:
            List of alert groups representing incidents
        """
        # Create subgraph with only strong edges
        strong_edges = [
            (u, v) for u, v, d in self.correlation_graph.edges(data=True)
            if d['weight'] >= min_weight
        ]
        
        subgraph = self.correlation_graph.edge_subgraph(strong_edges)
        
        # Find connected components
        components = list(nx.connected_components(subgraph))
        
        # Convert to alert groups
        alert_groups = []
        for component in components:
            alerts = [
                self.correlation_graph.nodes[alert_id]['alert']
                for alert_id in component
            ]
            alert_groups.append(alerts)
        
        return alert_groups
    
    def get_correlation_explanation(self, alert_id1: str, alert_id2: str) -> str:
        """Get explanation of why two alerts are correlated"""
        if self.correlation_graph.has_edge(alert_id1, alert_id2):
            edge_data = self.correlation_graph[alert_id1][alert_id2]
            return f"Correlation strength: {edge_data['weight']:.2f}. Reasons: {', '.join(edge_data['reasons'])}"
        return "Not correlated"


class IncidentAggregator:
    """
    Aggregates correlated alerts into coherent incidents with AI summarization.
    """
    
    def __init__(self, bedrock_client):
        """
        Initialize incident aggregator.
        
        Args:
            bedrock_client: AWS Bedrock client for summarization
        """
        self.bedrock_client = bedrock_client
        self.incidents: Dict[str, Incident] = {}
    
    def create_incident_from_alerts(self,
                                   alerts: List[Alert],
                                   correlation_confidence: float = 0.8) -> Incident:
        """
        Create an incident from a group of correlated alerts.
        
        Args:
            alerts: List of correlated alerts
            correlation_confidence: Confidence score of the correlation
            
        Returns:
            Incident object
        """
        if not alerts:
            raise ValueError("Cannot create incident from empty alert list")
        
        # Determine primary alert (highest severity, earliest timestamp)
        primary_alert = max(
            alerts,
            key=lambda a: (self._severity_to_score(a.severity), -a.timestamp.timestamp())
        )
        
        # Determine incident severity (highest among alerts)
        incident_severity = max(
            alerts,
            key=lambda a: self._severity_to_score(a.severity)
        ).severity
        
        # Collect affected entities
        affected_entities = set(a.affected_entity for a in alerts)
        
        # Generate incident title
        if len(alerts) == 1:
            title = alerts[0].title
        else:
            title = f"{primary_alert.title} (+{len(alerts)-1} related alerts)"
        
        # Generate AI summary
        summary = self._generate_incident_summary(alerts)
        
        incident_id = f"inc_{hashlib.md5(str(primary_alert.timestamp).encode()).hexdigest()[:8]}"
        
        incident = Incident(
            incident_id=incident_id,
            created_at=min(a.timestamp for a in alerts),
            updated_at=datetime.now(),
            alerts=alerts,
            primary_alert=primary_alert,
            severity=incident_severity,
            title=title,
            summary=summary,
            affected_entities=affected_entities,
            status='active',
            correlation_confidence=correlation_confidence
        )
        
        self.incidents[incident_id] = incident
        return incident
    
    def _severity_to_score(self, severity: str) -> int:
        """Convert severity to numeric score for comparison"""
        severity_map = {'critical': 3, 'warning': 2, 'info': 1}
        return severity_map.get(severity.lower(), 0)
    
    def _generate_incident_summary(self, alerts: List[Alert]) -> str:
        """
        Generate natural language summary of incident using AWS Bedrock.
        
        Args:
            alerts: Alerts in the incident
            
        Returns:
            Natural language summary
        """
        # Prepare alert information for LLM
        alert_info = []
        for i, alert in enumerate(alerts, 1):
            alert_info.append(
                f"{i}. [{alert.severity.upper()}] {alert.title} on {alert.affected_entity} "
                f"at {alert.timestamp.strftime('%H:%M:%S')}"
            )
        
        prompt = f"""Analyze these correlated alerts and provide a concise incident summary:

{chr(10).join(alert_info)}

Provide:
1. What is happening (2-3 sentences)
2. Likely root cause
3. Immediate impact
4. Recommended first step

Be specific and actionable."""

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
    
    def update_incident(self, incident_id: str, new_alerts: List[Alert]):
        """Add new alerts to an existing incident"""
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            incident.alerts.extend(new_alerts)
            incident.updated_at = datetime.now()
            incident.affected_entities.update(a.affected_entity for a in new_alerts)
            
            # Regenerate summary
            incident.summary = self._generate_incident_summary(incident.alerts)


class CorrelationPipeline:
    """
    Production correlation pipeline orchestrating all correlation approaches.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize correlation pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.deduplicator = AlertDeduplicator()
        self.topology_correlator = TopologyCorrelator()
        self.temporal_correlator = TemporalCorrelator()
        self.semantic_correlator = SemanticCorrelator(self.bedrock_client)
        self.graph_correlator = GraphCorrelator()
        self.incident_aggregator = IncidentAggregator(self.bedrock_client)
        
        self.alert_buffer: List[Alert] = []
        self.processing_window_minutes = 5
    
    def add_service_dependency(self, service: str, depends_on: str):
        """Add a known service dependency"""
        self.topology_correlator.add_dependency(service, depends_on)
    
    def process_alert(self, alert: Alert) -> Dict[str, Any]:
        """
        Process a single alert through the correlation pipeline.
        
        Args:
            alert: Alert to process
            
        Returns:
            Processing result with deduplication and correlation info
        """
        result = {
            'alert_id': alert.alert_id,
            'is_duplicate': False,
            'correlation_applied': False,
            'incident_id': None
        }
        
        # Step 1: Deduplication
        is_exact_dup, existing_id = self.deduplicator.exact_deduplicate(alert)
        
        if is_exact_dup:
            result['is_duplicate'] = True
            result['duplicate_of'] = existing_id
            result['dedup_type'] = 'exact'
            return result
        
        # Check fuzzy deduplication
        is_fuzzy_dup, existing_id, similarity = self.deduplicator.fuzzy_deduplicate(alert)
        
        if is_fuzzy_dup:
            result['is_duplicate'] = True
            result['duplicate_of'] = existing_id
            result['dedup_type'] = 'fuzzy'
            result['similarity_score'] = similarity
            return result
        
        # Check temporal deduplication (flapping)
        is_flapping, original_id = self.deduplicator.temporal_deduplicate(alert)
        
        if is_flapping:
            result['is_duplicate'] = True
            result['duplicate_of'] = original_id
            result['dedup_type'] = 'flapping'
            return result
        
        # Step 2: Add to buffer for correlation
        self.alert_buffer.append(alert)
        
        # Step 3: Periodically correlate buffered alerts
        if self._should_correlate():
            incidents = self.correlate_buffered_alerts()
            result['correlation_applied'] = True
            result['incidents_created'] = len(incidents)
            
            # Find which incident this alert belongs to
            for incident in incidents:
                if alert in incident.alerts:
                    result['incident_id'] = incident.incident_id
                    break
        
        return result
    
    def _should_correlate(self) -> bool:
        """Determine if it's time to correlate buffered alerts"""
        if not self.alert_buffer:
            return False
        
        # Correlate if we have enough alerts or enough time has passed
        time_since_oldest = (datetime.now() - 
                            min(a.timestamp for a in self.alert_buffer)).total_seconds() / 60
        
        return len(self.alert_buffer) >= 10 or time_since_oldest >= self.processing_window_minutes
    
    def correlate_buffered_alerts(self) -> List[Incident]:
        """
        Correlate all buffered alerts using multiple approaches.
        
        Returns:
            List of created incidents
        """
        if not self.alert_buffer:
            return []
        
        print(f"\nCorrelating {len(self.alert_buffer)} buffered alerts...")
        
        # Build correlation graph combining all signals
        correlators = {
            'temporal': self.temporal_correlator,
            'topology': self.topology_correlator,
            'semantic': self.semantic_correlator
        }
        
        self.graph_correlator.build_correlation_graph(self.alert_buffer, correlators)
        
        # Detect incident groups
        alert_groups = self.graph_correlator.detect_incident_groups(min_weight=0.5)
        
        print(f"Detected {len(alert_groups)} incident groups")
        
        # Create incidents from groups
        incidents = []
        for group in alert_groups:
            if len(group) > 0:  # Only create incidents for non-empty groups
                incident = self.incident_aggregator.create_incident_from_alerts(
                    group,
                    correlation_confidence=0.85
                )
                incidents.append(incident)
        
        # Clear buffer
        self.alert_buffer.clear()
        
        return incidents
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'deduplication': self.deduplicator.get_dedup_stats(),
            'buffered_alerts': len(self.alert_buffer),
            'active_incidents': len(self.incident_aggregator.incidents),
            'topology_nodes': self.topology_correlator.topology_graph.number_of_nodes(),
            'topology_edges': self.topology_correlator.topology_graph.number_of_edges()
        }


# Example usage and realistic data generators
def generate_sample_alerts(count: int = 50, scenario: str = 'cascade') -> List[Alert]:
    """
    Generate realistic sample alerts for testing.
    
    Args:
        count: Number of alerts to generate
        scenario: Type of scenario ('cascade', 'independent', 'flapping')
        
    Returns:
        List of sample alerts
    """
    alerts = []
    base_time = datetime.now() - timedelta(minutes=30)
    
    services = ['web-frontend', 'api-gateway', 'auth-service', 'user-service',
                'database-primary', 'cache-redis', 'payment-service']
    
    if scenario == 'cascade':
        # Simulate cascading failure starting from database
        # Database fails first
        alerts.append(Alert(
            alert_id=f"alert_001",
            timestamp=base_time,
            source='prometheus',
            severity='critical',
            title='Database Connection Pool Exhausted',
            message='Connection pool exhausted: all 100 connections in use',
            affected_entity='database-primary',
            tags=['database', 'critical', 'connection']
        ))
        
        # Services dependent on database start failing
        for i, service in enumerate(['auth-service', 'user-service', 'payment-service'], 2):
            alerts.append(Alert(
                alert_id=f"alert_{i:03d}",
                timestamp=base_time + timedelta(seconds=30*i),
                source='cloudwatch',
                severity='critical',
                title='Database Query Timeout',
                message=f'Queries to database timing out after 5000ms',
                affected_entity=service,
                tags=['database', 'timeout', service]
            ))
        
        # Frontend services show elevated errors
        for i, service in enumerate(['api-gateway', 'web-frontend'], 5):
            alerts.append(Alert(
                alert_id=f"alert_{i:03d}",
                timestamp=base_time + timedelta(minutes=2),
                source='cloudwatch',
                severity='warning',
                title='Elevated Error Rate',
                message=f'Error rate increased to 15% (baseline: 0.1%)',
                affected_entity=service,
                tags=['errors', 'degraded', service]
            ))
        
    elif scenario == 'flapping':
        # Simulate flapping alert
        service = 'cache-redis'
        for i in range(5):
            alerts.append(Alert(
                alert_id=f"alert_{i:03d}",
                timestamp=base_time + timedelta(minutes=i*3),
                source='prometheus',
                severity='warning',
                title='High Memory Usage',
                message='Memory usage at 85%',
                affected_entity=service,
                tags=['memory', 'redis']
            ))
    
    else:  # independent
        # Generate independent alerts
        for i in range(count):
            service = np.random.choice(services)
            severity = np.random.choice(['critical', 'warning', 'info'], p=[0.2, 0.5, 0.3])
            
            alerts.append(Alert(
                alert_id=f"alert_{i:03d}",
                timestamp=base_time + timedelta(minutes=np.random.randint(0, 30)),
                source=np.random.choice(['prometheus', 'cloudwatch', 'datadog']),
                severity=severity,
                title=np.random.choice(['High CPU', 'High Memory', 'Slow Response']),
                message=f'Alert on {service}',
                affected_entity=service,
                tags=[service, severity]
            ))
    
    return sorted(alerts, key=lambda a: a.timestamp)


def main():
    """
    Demonstrate correlation and deduplication pipeline with realistic scenarios.
    """
    print("=" * 80)
    print("Chapter 11: Correlation and Deduplication")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing correlation pipeline...")
    pipeline = CorrelationPipeline(aws_region='us-east-1')
    
    # Define service topology
    print("Setting up service topology...")
    pipeline.add_service_dependency('web-frontend', 'api-gateway')
    pipeline.add_service_dependency('api-gateway', 'auth-service')
    pipeline.add_service_dependency('api-gateway', 'user-service')
    pipeline.add_service_dependency('auth-service', 'database-primary')
    pipeline.add_service_dependency('user-service', 'database-primary')
    pipeline.add_service_dependency('user-service', 'cache-redis')
    pipeline.add_service_dependency('payment-service', 'database-primary')
    print()
    
    # Scenario 1: Cascading Failure
    print("=" * 80)
    print("SCENARIO 1: Cascading Database Failure")
    print("=" * 80)
    print()
    
    cascade_alerts = generate_sample_alerts(count=20, scenario='cascade')
    print(f"Processing {len(cascade_alerts)} alerts from cascading failure...")
    print()
    
    for alert in cascade_alerts:
        result = pipeline.process_alert(alert)
        
        if result['is_duplicate']:
            print(f"✓ Alert {alert.alert_id}: DUPLICATE ({result['dedup_type']})")
        else:
            print(f"→ Alert {alert.alert_id}: {alert.severity.upper()} - {alert.title} on {alert.affected_entity}")
    
    # Force correlation
    incidents = pipeline.correlate_buffered_alerts()
    
    print()
    print(f"Correlation Complete: {len(incidents)} incident(s) created")
    print("-" * 80)
    
    for incident in incidents:
        print(f"\n📋 INCIDENT: {incident.incident_id}")
        print(f"   Severity: {incident.severity.upper()}")
        print(f"   Title: {incident.title}")
        print(f"   Alerts: {len(incident.alerts)}")
        print(f"   Affected Entities: {', '.join(incident.affected_entities)}")
        print(f"   Correlation Confidence: {incident.correlation_confidence:.2%}")
        print(f"\n   Summary:")
        print(f"   {incident.summary}")
    
    # Scenario 2: Flapping Alert
    print("\n" + "=" * 80)
    print("SCENARIO 2: Flapping Alert Detection")
    print("=" * 80)
    print()
    
    # Reset pipeline for clean test
    pipeline = CorrelationPipeline(aws_region='us-east-1')
    
    flapping_alerts = generate_sample_alerts(count=10, scenario='flapping')
    print(f"Processing {len(flapping_alerts)} potentially flapping alerts...")
    print()
    
    for alert in flapping_alerts:
        result = pipeline.process_alert(alert)
        
        if result['is_duplicate']:
            print(f"✓ Alert {alert.alert_id}: FLAPPING (duplicate of {result['duplicate_of']})")
        else:
            print(f"→ Alert {alert.alert_id}: {alert.severity.upper()} - {alert.title}")
            # Simulate some alerts resolving
            if np.random.random() > 0.5:
                pipeline.deduplicator.resolve_alert(
                    alert.alert_id,
                    alert.timestamp + timedelta(minutes=1)
                )
    
    # Show statistics
    print("\n" + "=" * 80)
    print("PIPELINE STATISTICS")
    print("=" * 80)
    
    stats = pipeline.get_pipeline_stats()
    
    print(f"\nDeduplication Stats:")
    print(f"  Total Alerts Seen: {stats['deduplication']['total_alerts_seen']}")
    print(f"  Unique Alerts: {stats['deduplication']['unique_alerts']}")
    print(f"  Duplicates Removed: {stats['deduplication']['duplicate_count']}")
    print(f"  Deduplication Rate: {stats['deduplication']['dedup_rate']:.1%}")
    
    print(f"\nCorrelation Stats:")
    print(f"  Active Incidents: {stats['active_incidents']}")
    print(f"  Topology Nodes: {stats['topology_nodes']}")
    print(f"  Topology Edges: {stats['topology_edges']}")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
