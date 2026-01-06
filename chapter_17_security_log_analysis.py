"""
Chapter 17: Security Log Analysis
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready AI-powered security log analysis including:
- Intelligent log parsing (template mining, LLM-based extraction)
- Entity extraction and normalization
- Cross-source log correlation (entity, temporal, graph-based)
- Threat hunting queries and pattern matching
- Investigation timeline reconstruction
- Evidence collection and chain of custody
- AI-powered log analysis with AWS Bedrock

Author: AI-Powered Observability
"""

import json
import boto3
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import warnings
warnings.filterwarnings('ignore')


class LogSource(Enum):
    """Types of security log sources"""
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    ENDPOINT = "endpoint"
    CLOUD = "cloud"
    APPLICATION = "application"
    DNS = "dns"
    FIREWALL = "firewall"


class EventSeverity(Enum):
    """Normalized event severity"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class RawLogEntry:
    """Represents a raw log entry before parsing"""
    timestamp: datetime
    source: str
    source_type: LogSource
    raw_message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedLogEntry:
    """Represents a parsed and normalized log entry"""
    log_id: str
    timestamp: datetime
    source: str
    source_type: LogSource
    event_type: str
    severity: EventSeverity
    entities: Dict[str, Any]  # Extracted entities (users, IPs, hosts, etc.)
    fields: Dict[str, Any]  # Normalized fields
    raw_message: str
    template: str  # Extracted log template
    hash: str  # Content hash for deduplication


@dataclass 
class InvestigationTimeline:
    """Represents a security investigation timeline"""
    investigation_id: str
    title: str
    start_time: datetime
    end_time: datetime
    entities_of_interest: List[str]
    events: List[ParsedLogEntry]
    findings: List[str]
    evidence_hashes: List[str]


@dataclass
class ThreatHuntQuery:
    """Represents a threat hunting query"""
    query_id: str
    name: str
    description: str
    mitre_technique: Optional[str]
    entity_filters: Dict[str, Any]
    field_filters: Dict[str, Any]
    time_window_hours: int


@dataclass
class HuntingResult:
    """Result from a threat hunting query"""
    query: ThreatHuntQuery
    matching_events: List[ParsedLogEntry]
    entity_summary: Dict[str, int]
    risk_score: float
    ai_analysis: str


class LogTemplateExtractor:
    """
    Extracts log templates using pattern mining (simplified Drain algorithm).
    """
    
    def __init__(self):
        """Initialize template extractor"""
        self.templates: Dict[str, str] = {}
        self.template_counts: Counter = Counter()
        
        # Common patterns to replace
        self.patterns = [
            (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>'),
            (r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>'),
            (r'[A-Fa-f0-9]{32,64}', '<HASH>'),
            (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '<EMAIL>'),
            (r'user[=: ]+[a-zA-Z0-9_-]+', 'user=<USER>'),
            (r'port[=: ]+\d+', 'port=<PORT>'),
            (r'\b[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*', '<FILEPATH>'),
            (r'/(?:[a-zA-Z0-9._-]+/)+[a-zA-Z0-9._-]+', '<FILEPATH>'),
        ]
    
    def extract_template(self, log_message: str) -> str:
        """
        Extract template from log message by replacing variables with placeholders.
        
        Args:
            log_message: Raw log message
            
        Returns:
            Template with variables replaced
        """
        template = log_message
        
        # Apply pattern replacements
        for pattern, replacement in self.patterns:
            template = re.sub(pattern, replacement, template)
        
        # Replace numbers that look like IDs or counts
        template = re.sub(r'\b\d{4,}\b', '<ID>', template)
        
        # Track template
        template_key = hashlib.md5(template.encode()).hexdigest()[:8]
        self.templates[template_key] = template
        self.template_counts[template_key] += 1
        
        return template
    
    def get_rare_templates(self, threshold: int = 5) -> List[str]:
        """Get templates that appear rarely (potential anomalies)"""
        return [
            self.templates[key] 
            for key, count in self.template_counts.items()
            if count <= threshold
        ]


class SecurityLogParser:
    """
    AI-powered security log parser using LLM for field extraction.
    """
    
    def __init__(self, bedrock_client):
        """Initialize security log parser"""
        self.bedrock_client = bedrock_client
        self.template_extractor = LogTemplateExtractor()
        self.entity_patterns = {
            'ip_address': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            'username': r'(?:user|username|account)[=:\s]+([a-zA-Z0-9_.-]+)',
            'hostname': r'(?:host|hostname|computer)[=:\s]+([a-zA-Z0-9_.-]+)',
            'filepath': r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]+|/(?:[a-zA-Z0-9._-]+/)+[a-zA-Z0-9._-]+',
            'process': r'(?:process|proc)[=:\s]+([a-zA-Z0-9_.-]+(?:\.exe)?)',
            'port': r'(?:port|dport|sport)[=:\s]+(\d+)',
        }
    
    def parse_log(self, raw_log: RawLogEntry) -> ParsedLogEntry:
        """
        Parse a raw log entry into structured format.
        
        Args:
            raw_log: Raw log entry
            
        Returns:
            Parsed and normalized log entry
        """
        # Extract template
        template = self.template_extractor.extract_template(raw_log.raw_message)
        
        # Extract entities
        entities = self._extract_entities(raw_log.raw_message)
        
        # Normalize fields based on source type
        fields = self._normalize_fields(raw_log)
        
        # Determine event type and severity
        event_type, severity = self._classify_event(raw_log.raw_message, raw_log.source_type)
        
        # Generate log ID and hash
        log_id = f"log_{raw_log.timestamp.timestamp()}_{hash(raw_log.raw_message) % 10000}"
        content_hash = hashlib.sha256(raw_log.raw_message.encode()).hexdigest()
        
        return ParsedLogEntry(
            log_id=log_id,
            timestamp=raw_log.timestamp,
            source=raw_log.source,
            source_type=raw_log.source_type,
            event_type=event_type,
            severity=severity,
            entities=entities,
            fields=fields,
            raw_message=raw_log.raw_message,
            template=template,
            hash=content_hash
        )
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from log message"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                # For patterns with groups, get the group
                if isinstance(matches[0], tuple):
                    matches = [m[0] if isinstance(m, tuple) else m for m in matches]
                entities[entity_type] = list(set(matches))
        
        # Extract all IPs directly
        all_ips = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', message)
        if all_ips:
            entities['ip_addresses'] = list(set(all_ips))
        
        return entities
    
    def _normalize_fields(self, raw_log: RawLogEntry) -> Dict[str, Any]:
        """Normalize fields based on source type"""
        fields = {
            'source_system': raw_log.source,
            'log_source_type': raw_log.source_type.value
        }
        
        # Extract common fields from message
        msg = raw_log.raw_message.lower()
        
        # Action/result
        if 'success' in msg or 'allowed' in msg or 'accepted' in msg:
            fields['action_result'] = 'success'
        elif 'fail' in msg or 'denied' in msg or 'blocked' in msg or 'rejected' in msg:
            fields['action_result'] = 'failure'
        
        # Direction for network logs
        if raw_log.source_type == LogSource.NETWORK or raw_log.source_type == LogSource.FIREWALL:
            if 'inbound' in msg or 'in=' in msg:
                fields['direction'] = 'inbound'
            elif 'outbound' in msg or 'out=' in msg:
                fields['direction'] = 'outbound'
        
        return fields
    
    def _classify_event(self, message: str, source_type: LogSource) -> Tuple[str, EventSeverity]:
        """Classify event type and severity"""
        msg = message.lower()
        
        # Authentication events
        if source_type == LogSource.AUTHENTICATION:
            if 'login' in msg or 'logon' in msg or 'auth' in msg:
                if 'fail' in msg:
                    return 'authentication.failure', EventSeverity.MEDIUM
                else:
                    return 'authentication.success', EventSeverity.INFO
            if 'logout' in msg or 'logoff' in msg:
                return 'authentication.logout', EventSeverity.INFO
            if 'password' in msg and 'change' in msg:
                return 'authentication.password_change', EventSeverity.LOW
        
        # Network events
        if source_type in [LogSource.NETWORK, LogSource.FIREWALL]:
            if 'block' in msg or 'denied' in msg or 'drop' in msg:
                return 'network.blocked', EventSeverity.LOW
            if 'allow' in msg or 'accept' in msg:
                return 'network.allowed', EventSeverity.INFO
            if 'scan' in msg:
                return 'network.scan', EventSeverity.MEDIUM
        
        # Endpoint events
        if source_type == LogSource.ENDPOINT:
            if 'process' in msg and 'creat' in msg:
                return 'endpoint.process_create', EventSeverity.INFO
            if 'malware' in msg or 'virus' in msg:
                return 'endpoint.malware_detected', EventSeverity.CRITICAL
            if 'registry' in msg:
                return 'endpoint.registry_change', EventSeverity.LOW
        
        # Cloud events  
        if source_type == LogSource.CLOUD:
            if 'createuser' in msg or 'create user' in msg:
                return 'cloud.user_created', EventSeverity.MEDIUM
            if 'delete' in msg:
                return 'cloud.resource_deleted', EventSeverity.MEDIUM
            if 'policy' in msg and 'change' in msg:
                return 'cloud.policy_change', EventSeverity.HIGH
        
        return 'unknown', EventSeverity.INFO
    
    def parse_with_llm(self, raw_log: RawLogEntry) -> ParsedLogEntry:
        """
        Use LLM for intelligent log parsing when standard parsing is insufficient.
        
        Args:
            raw_log: Raw log entry
            
        Returns:
            Parsed log entry with LLM-extracted fields
        """
        prompt = f"""Parse this security log entry and extract structured information:

Log: {raw_log.raw_message}

Extract the following if present:
1. Event type (authentication, network, file access, etc.)
2. Action and result (success/failure)
3. Source IP and destination IP
4. Username or account
5. Hostname or system
6. Any file paths or processes
7. Severity (critical/high/medium/low/info)

Respond in JSON format with these fields:
{{
    "event_type": "",
    "action_result": "",
    "source_ip": "",
    "dest_ip": "",
    "username": "",
    "hostname": "",
    "filepath": "",
    "process": "",
    "severity": "",
    "summary": ""
}}"""

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
            llm_output = result['content'][0]['text']
            
            # Parse JSON from LLM response
            json_match = re.search(r'\{[^{}]+\}', llm_output, re.DOTALL)
            if json_match:
                parsed_fields = json.loads(json_match.group())
                
                # Build entities from LLM output
                entities = {}
                if parsed_fields.get('source_ip'):
                    entities['source_ip'] = [parsed_fields['source_ip']]
                if parsed_fields.get('dest_ip'):
                    entities['dest_ip'] = [parsed_fields['dest_ip']]
                if parsed_fields.get('username'):
                    entities['username'] = [parsed_fields['username']]
                if parsed_fields.get('hostname'):
                    entities['hostname'] = [parsed_fields['hostname']]
                
                # Map severity
                severity_map = {
                    'critical': EventSeverity.CRITICAL,
                    'high': EventSeverity.HIGH,
                    'medium': EventSeverity.MEDIUM,
                    'low': EventSeverity.LOW,
                    'info': EventSeverity.INFO
                }
                severity = severity_map.get(
                    parsed_fields.get('severity', '').lower(),
                    EventSeverity.INFO
                )
                
                return ParsedLogEntry(
                    log_id=f"log_{raw_log.timestamp.timestamp()}_{hash(raw_log.raw_message) % 10000}",
                    timestamp=raw_log.timestamp,
                    source=raw_log.source,
                    source_type=raw_log.source_type,
                    event_type=parsed_fields.get('event_type', 'unknown'),
                    severity=severity,
                    entities=entities,
                    fields=parsed_fields,
                    raw_message=raw_log.raw_message,
                    template=self.template_extractor.extract_template(raw_log.raw_message),
                    hash=hashlib.sha256(raw_log.raw_message.encode()).hexdigest()
                )
        
        except Exception as e:
            print(f"LLM parsing failed: {e}")
        
        # Fallback to standard parsing
        return self.parse_log(raw_log)


class LogCorrelator:
    """
    Correlates logs across sources using entity, temporal, and graph-based methods.
    """
    
    def __init__(self):
        """Initialize log correlator"""
        self.logs_by_entity: Dict[str, List[ParsedLogEntry]] = defaultdict(list)
        self.logs_by_time: List[ParsedLogEntry] = []
        self.entity_graph: Dict[str, Set[str]] = defaultdict(set)  # Entity relationships
    
    def add_log(self, log: ParsedLogEntry):
        """Add log to correlation indexes"""
        # Index by entities
        for entity_type, values in log.entities.items():
            for value in values:
                entity_key = f"{entity_type}:{value}"
                self.logs_by_entity[entity_key].append(log)
        
        # Add to time-sorted list
        self.logs_by_time.append(log)
        self.logs_by_time.sort(key=lambda l: l.timestamp)
        
        # Build entity graph (entities in same log are related)
        entity_keys = []
        for entity_type, values in log.entities.items():
            for value in values:
                entity_keys.append(f"{entity_type}:{value}")
        
        for i, key1 in enumerate(entity_keys):
            for key2 in entity_keys[i+1:]:
                self.entity_graph[key1].add(key2)
                self.entity_graph[key2].add(key1)
    
    def correlate_by_entity(self, entity_type: str, entity_value: str) -> List[ParsedLogEntry]:
        """
        Get all logs involving a specific entity.
        
        Args:
            entity_type: Type of entity (username, ip_address, etc.)
            entity_value: Entity value to search for
            
        Returns:
            List of correlated logs
        """
        entity_key = f"{entity_type}:{entity_value}"
        return sorted(self.logs_by_entity.get(entity_key, []), key=lambda l: l.timestamp)
    
    def correlate_by_time_window(self,
                                 center_time: datetime,
                                 window_seconds: int = 300) -> List[ParsedLogEntry]:
        """
        Get all logs within a time window.
        
        Args:
            center_time: Center of time window
            window_seconds: Window size in seconds (default 5 minutes)
            
        Returns:
            Logs within the time window
        """
        start = center_time - timedelta(seconds=window_seconds/2)
        end = center_time + timedelta(seconds=window_seconds/2)
        
        return [
            log for log in self.logs_by_time
            if start <= log.timestamp <= end
        ]
    
    def find_related_entities(self, entity_type: str, entity_value: str, depth: int = 2) -> Set[str]:
        """
        Find entities related through the entity graph.
        
        Args:
            entity_type: Type of entity
            entity_value: Entity value
            depth: How many hops to traverse
            
        Returns:
            Set of related entity keys
        """
        entity_key = f"{entity_type}:{entity_value}"
        related = set()
        current_level = {entity_key}
        
        for _ in range(depth):
            next_level = set()
            for entity in current_level:
                for neighbor in self.entity_graph.get(entity, set()):
                    if neighbor not in related and neighbor != entity_key:
                        next_level.add(neighbor)
                        related.add(neighbor)
            current_level = next_level
        
        return related
    
    def build_activity_timeline(self,
                               entity_type: str,
                               entity_value: str,
                               start_time: datetime,
                               end_time: datetime) -> List[ParsedLogEntry]:
        """
        Build complete activity timeline for an entity.
        
        Args:
            entity_type: Type of entity
            entity_value: Entity value
            start_time: Timeline start
            end_time: Timeline end
            
        Returns:
            Chronological list of events
        """
        entity_logs = self.correlate_by_entity(entity_type, entity_value)
        
        return [
            log for log in entity_logs
            if start_time <= log.timestamp <= end_time
        ]


class ThreatHunter:
    """
    Executes threat hunting queries against parsed logs.
    """
    
    def __init__(self, bedrock_client, correlator: LogCorrelator):
        """Initialize threat hunter"""
        self.bedrock_client = bedrock_client
        self.correlator = correlator
        self.predefined_hunts = self._create_predefined_hunts()
    
    def _create_predefined_hunts(self) -> List[ThreatHuntQuery]:
        """Create predefined threat hunting queries"""
        return [
            ThreatHuntQuery(
                query_id='hunt_001',
                name='Brute Force Authentication',
                description='Detect multiple failed authentication attempts',
                mitre_technique='T1110',
                entity_filters={},
                field_filters={'action_result': 'failure'},
                time_window_hours=1
            ),
            ThreatHuntQuery(
                query_id='hunt_002',
                name='Suspicious PowerShell Execution',
                description='Detect encoded or obfuscated PowerShell commands',
                mitre_technique='T1059.001',
                entity_filters={'process': ['powershell.exe', 'pwsh.exe']},
                field_filters={},
                time_window_hours=24
            ),
            ThreatHuntQuery(
                query_id='hunt_003',
                name='Lateral Movement Detection',
                description='Detect authentication to multiple hosts from single source',
                mitre_technique='T1021',
                entity_filters={},
                field_filters={'action_result': 'success'},
                time_window_hours=4
            ),
            ThreatHuntQuery(
                query_id='hunt_004',
                name='Data Exfiltration Indicators',
                description='Detect large outbound data transfers',
                mitre_technique='T1048',
                entity_filters={},
                field_filters={'direction': 'outbound'},
                time_window_hours=24
            ),
            ThreatHuntQuery(
                query_id='hunt_005',
                name='Persistence Mechanism',
                description='Detect registry modifications for persistence',
                mitre_technique='T1547',
                entity_filters={},
                field_filters={},
                time_window_hours=72
            )
        ]
    
    def execute_hunt(self,
                    query: ThreatHuntQuery,
                    logs: List[ParsedLogEntry]) -> HuntingResult:
        """
        Execute a threat hunting query.
        
        Args:
            query: Threat hunting query
            logs: Logs to search
            
        Returns:
            Hunting result with matches and analysis
        """
        matching_events = []
        
        for log in logs:
            match = True
            
            # Check entity filters
            for entity_type, allowed_values in query.entity_filters.items():
                log_values = log.entities.get(entity_type, [])
                if not any(v.lower() in [av.lower() for av in allowed_values] for v in log_values):
                    match = False
                    break
            
            # Check field filters
            if match:
                for field_name, expected_value in query.field_filters.items():
                    actual_value = log.fields.get(field_name, '')
                    if isinstance(expected_value, str):
                        if expected_value.lower() not in str(actual_value).lower():
                            match = False
                            break
            
            if match:
                matching_events.append(log)
        
        # Calculate entity summary
        entity_summary = Counter()
        for event in matching_events:
            for entity_type, values in event.entities.items():
                for value in values:
                    entity_summary[f"{entity_type}:{value}"] += 1
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(query, matching_events)
        
        # Generate AI analysis
        ai_analysis = self._generate_hunt_analysis(query, matching_events, entity_summary)
        
        return HuntingResult(
            query=query,
            matching_events=matching_events,
            entity_summary=dict(entity_summary),
            risk_score=risk_score,
            ai_analysis=ai_analysis
        )
    
    def _calculate_risk_score(self,
                             query: ThreatHuntQuery,
                             events: List[ParsedLogEntry]) -> float:
        """Calculate risk score based on hunting results"""
        if not events:
            return 0.0
        
        score = 0.0
        
        # Volume score (more events = higher risk for some hunts)
        if query.query_id == 'hunt_001':  # Brute force
            score = min(len(events) / 10, 1.0)
        elif query.query_id == 'hunt_002':  # PowerShell
            score = min(len(events) / 5, 1.0)
        elif query.query_id == 'hunt_003':  # Lateral movement
            unique_hosts = len(set(
                v for e in events 
                for v in e.entities.get('hostname', [])
            ))
            score = min(unique_hosts / 5, 1.0)
        else:
            score = min(len(events) / 20, 1.0)
        
        # Boost for critical severity events
        critical_count = sum(1 for e in events if e.severity == EventSeverity.CRITICAL)
        score = min(score + (critical_count * 0.2), 1.0)
        
        return score
    
    def _generate_hunt_analysis(self,
                               query: ThreatHuntQuery,
                               events: List[ParsedLogEntry],
                               entity_summary: Dict[str, int]) -> str:
        """Generate AI analysis of hunting results"""
        if not events:
            return "No matching events found for this hunt."
        
        top_entities = sorted(entity_summary.items(), key=lambda x: x[1], reverse=True)[:5]
        entities_text = "\n".join([f"  - {k}: {v} occurrences" for k, v in top_entities])
        
        sample_events = "\n".join([
            f"  - [{e.timestamp}] {e.event_type}: {e.raw_message[:100]}..."
            for e in events[:3]
        ])
        
        prompt = f"""Analyze these threat hunting results:

Hunt: {query.name}
Description: {query.description}
MITRE Technique: {query.mitre_technique}

Results:
- Total matching events: {len(events)}
- Top entities:
{entities_text}

Sample events:
{sample_events}

Provide a brief (3-4 sentences) security analysis:
1. What the results indicate
2. Whether this appears to be malicious activity
3. Recommended next steps for investigation"""

        try:
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
        
        except Exception as e:
            return f"Analysis unavailable: {e}"
    
    def run_all_hunts(self, logs: List[ParsedLogEntry]) -> List[HuntingResult]:
        """Run all predefined threat hunts"""
        results = []
        
        for hunt in self.predefined_hunts:
            result = self.execute_hunt(hunt, logs)
            results.append(result)
        
        return results


class InvestigationWorkbench:
    """
    Supports security investigations with timeline reconstruction and evidence collection.
    """
    
    def __init__(self, bedrock_client, correlator: LogCorrelator):
        """Initialize investigation workbench"""
        self.bedrock_client = bedrock_client
        self.correlator = correlator
        self.investigations: Dict[str, InvestigationTimeline] = {}
    
    def create_investigation(self,
                           title: str,
                           entities: List[str],
                           start_time: datetime,
                           end_time: datetime) -> InvestigationTimeline:
        """
        Create a new investigation.
        
        Args:
            title: Investigation title
            entities: Entities of interest (format: "type:value")
            start_time: Investigation start time
            end_time: Investigation end time
            
        Returns:
            New investigation timeline
        """
        investigation_id = f"inv_{datetime.now().timestamp()}"
        
        # Collect all relevant events
        all_events = []
        for entity in entities:
            if ':' in entity:
                entity_type, entity_value = entity.split(':', 1)
                events = self.correlator.build_activity_timeline(
                    entity_type, entity_value, start_time, end_time
                )
                all_events.extend(events)
        
        # Deduplicate and sort
        seen_ids = set()
        unique_events = []
        for event in sorted(all_events, key=lambda e: e.timestamp):
            if event.log_id not in seen_ids:
                seen_ids.add(event.log_id)
                unique_events.append(event)
        
        investigation = InvestigationTimeline(
            investigation_id=investigation_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            entities_of_interest=entities,
            events=unique_events,
            findings=[],
            evidence_hashes=[e.hash for e in unique_events]
        )
        
        self.investigations[investigation_id] = investigation
        
        return investigation
    
    def generate_investigation_report(self, investigation_id: str) -> str:
        """Generate AI-powered investigation report"""
        if investigation_id not in self.investigations:
            return "Investigation not found"
        
        inv = self.investigations[investigation_id]
        
        # Summarize events by type
        event_types = Counter(e.event_type for e in inv.events)
        severity_counts = Counter(e.severity.value for e in inv.events)
        
        events_summary = "\n".join([
            f"  - {etype}: {count}" for etype, count in event_types.most_common(10)
        ])
        
        timeline_summary = ""
        if inv.events:
            first_event = inv.events[0]
            last_event = inv.events[-1]
            timeline_summary = f"""
First event: {first_event.timestamp} - {first_event.event_type}
Last event: {last_event.timestamp} - {last_event.event_type}
Total events: {len(inv.events)}
Duration: {(last_event.timestamp - first_event.timestamp).total_seconds() / 3600:.1f} hours"""
        
        prompt = f"""Generate a security investigation report:

Investigation: {inv.title}
Entities of Interest: {', '.join(inv.entities_of_interest)}
Time Range: {inv.start_time} to {inv.end_time}

Timeline Summary:{timeline_summary}

Event Types:
{events_summary}

Severity Distribution:
{json.dumps(dict(severity_counts), indent=2)}

Provide a structured investigation report covering:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Timeline of Significant Events
4. Recommended Actions

Be specific and actionable."""

        try:
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 800,
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


class SecurityLogAnalysisPipeline:
    """
    Production pipeline orchestrating all security log analysis components.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize security log analysis pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.parser = SecurityLogParser(self.bedrock_client)
        self.correlator = LogCorrelator()
        self.hunter = ThreatHunter(self.bedrock_client, self.correlator)
        self.workbench = InvestigationWorkbench(self.bedrock_client, self.correlator)
        
        self.parsed_logs: List[ParsedLogEntry] = []
    
    def ingest_logs(self, raw_logs: List[RawLogEntry]) -> List[ParsedLogEntry]:
        """
        Ingest and parse raw logs.
        
        Args:
            raw_logs: Raw log entries
            
        Returns:
            Parsed log entries
        """
        parsed = []
        
        for raw_log in raw_logs:
            parsed_log = self.parser.parse_log(raw_log)
            parsed.append(parsed_log)
            self.correlator.add_log(parsed_log)
        
        self.parsed_logs.extend(parsed)
        
        return parsed
    
    def run_threat_hunts(self) -> List[HuntingResult]:
        """Run all threat hunting queries"""
        return self.hunter.run_all_hunts(self.parsed_logs)
    
    def investigate_entity(self,
                         entity_type: str,
                         entity_value: str,
                         hours_back: int = 24) -> InvestigationTimeline:
        """
        Start investigation for an entity.
        
        Args:
            entity_type: Type of entity
            entity_value: Entity value
            hours_back: How far back to investigate
            
        Returns:
            Investigation timeline
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        return self.workbench.create_investigation(
            title=f"Investigation: {entity_type}={entity_value}",
            entities=[f"{entity_type}:{entity_value}"],
            start_time=start_time,
            end_time=end_time
        )
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get log analysis statistics"""
        if not self.parsed_logs:
            return {'no_logs': True}
        
        by_source = Counter(l.source_type.value for l in self.parsed_logs)
        by_severity = Counter(l.severity.value for l in self.parsed_logs)
        by_event_type = Counter(l.event_type for l in self.parsed_logs)
        
        # Unique entities
        all_entities = set()
        for log in self.parsed_logs:
            for entity_type, values in log.entities.items():
                for value in values:
                    all_entities.add(f"{entity_type}:{value}")
        
        return {
            'total_logs': len(self.parsed_logs),
            'by_source': dict(by_source),
            'by_severity': dict(by_severity),
            'top_event_types': dict(by_event_type.most_common(10)),
            'unique_entities': len(all_entities),
            'time_range': {
                'first': min(l.timestamp for l in self.parsed_logs).isoformat(),
                'last': max(l.timestamp for l in self.parsed_logs).isoformat()
            }
        }


# Example usage and realistic log samples
def generate_sample_logs() -> List[RawLogEntry]:
    """Generate realistic security log samples"""
    logs = []
    base_time = datetime.now() - timedelta(hours=6)
    
    # Authentication logs
    auth_logs = [
        "2024-03-15 10:15:23 auth-server sshd[12345]: Accepted publickey for admin from 192.168.1.50 port 52431 ssh2",
        "2024-03-15 10:16:45 auth-server sshd[12346]: Failed password for invalid user root from 203.0.113.50 port 45123 ssh2",
        "2024-03-15 10:16:46 auth-server sshd[12346]: Failed password for invalid user root from 203.0.113.50 port 45124 ssh2",
        "2024-03-15 10:16:47 auth-server sshd[12346]: Failed password for invalid user admin from 203.0.113.50 port 45125 ssh2",
        "2024-03-15 10:20:00 dc01 Security[4624]: An account was successfully logged on. Subject: CORP\\jsmith Logon Type: 3 Source IP: 192.168.1.100",
        "2024-03-15 10:25:00 dc01 Security[4625]: An account failed to log on. Subject: CORP\\admin Source IP: 10.0.0.50 Failure Reason: Bad password",
    ]
    
    for i, msg in enumerate(auth_logs):
        logs.append(RawLogEntry(
            timestamp=base_time + timedelta(minutes=i*5),
            source='auth-server',
            source_type=LogSource.AUTHENTICATION,
            raw_message=msg
        ))
    
    # Firewall logs
    firewall_logs = [
        "Mar 15 10:30:00 firewall01 kernel: [UFW BLOCK] IN=eth0 OUT= MAC=00:11:22:33:44:55 SRC=203.0.113.100 DST=192.168.1.10 PROTO=TCP SPT=45678 DPT=22",
        "Mar 15 10:31:00 firewall01 kernel: [UFW ALLOW] IN=eth0 OUT= SRC=192.168.1.50 DST=8.8.8.8 PROTO=UDP SPT=54321 DPT=53",
        "Mar 15 10:32:00 firewall01 kernel: [UFW BLOCK] IN=eth0 OUT= SRC=185.143.223.47 DST=192.168.1.10 PROTO=TCP SPT=12345 DPT=4444",
    ]
    
    for i, msg in enumerate(firewall_logs):
        logs.append(RawLogEntry(
            timestamp=base_time + timedelta(minutes=30+i),
            source='firewall01',
            source_type=LogSource.FIREWALL,
            raw_message=msg
        ))
    
    # Endpoint logs
    endpoint_logs = [
        "2024-03-15 10:35:00 WORKSTATION-01 Process Create: Process Name: powershell.exe Parent Process: excel.exe Command Line: powershell.exe -encodedcommand SQBFAFgA User: CORP\\jsmith",
        "2024-03-15 10:36:00 WORKSTATION-01 Registry Modification: Key: HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run Value: UpdateCheck Data: C:\\Users\\jsmith\\update.exe",
        "2024-03-15 10:37:00 SERVER-DB01 Process Create: Process Name: cmd.exe Parent Process: sqlservr.exe Command Line: cmd.exe /c whoami User: NT AUTHORITY\\SYSTEM",
    ]
    
    for i, msg in enumerate(endpoint_logs):
        logs.append(RawLogEntry(
            timestamp=base_time + timedelta(minutes=35+i),
            source='edr-agent',
            source_type=LogSource.ENDPOINT,
            raw_message=msg
        ))
    
    # Cloud audit logs
    cloud_logs = [
        '{"eventTime": "2024-03-15T10:40:00Z", "eventName": "ConsoleLogin", "sourceIPAddress": "198.51.100.50", "userIdentity": {"userName": "admin@company.com"}, "responseElements": {"ConsoleLogin": "Success"}}',
        '{"eventTime": "2024-03-15T10:42:00Z", "eventName": "CreateUser", "sourceIPAddress": "198.51.100.50", "userIdentity": {"userName": "admin@company.com"}, "requestParameters": {"userName": "backdoor-user"}}',
        '{"eventTime": "2024-03-15T10:45:00Z", "eventName": "AttachUserPolicy", "sourceIPAddress": "198.51.100.50", "requestParameters": {"userName": "backdoor-user", "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess"}}',
    ]
    
    for i, msg in enumerate(cloud_logs):
        logs.append(RawLogEntry(
            timestamp=base_time + timedelta(minutes=40+i*2),
            source='cloudtrail',
            source_type=LogSource.CLOUD,
            raw_message=msg
        ))
    
    # DNS logs
    dns_logs = [
        "15-Mar-2024 10:50:00.123 client 192.168.1.100#54321: query: google.com IN A",
        "15-Mar-2024 10:50:01.456 client 192.168.1.100#54322: query: x7k9m2p.malware-c2.net IN A",
        "15-Mar-2024 10:50:02.789 client 192.168.1.100#54323: query: aHR0cHM6Ly9ldmlsLmNvbQ.tunnel.attacker.net IN TXT",
    ]
    
    for i, msg in enumerate(dns_logs):
        logs.append(RawLogEntry(
            timestamp=base_time + timedelta(minutes=50+i),
            source='dns-server',
            source_type=LogSource.DNS,
            raw_message=msg
        ))
    
    return logs


def main():
    """
    Demonstrate security log analysis pipeline with realistic scenarios.
    """
    print("=" * 80)
    print("Chapter 17: Security Log Analysis")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing security log analysis pipeline...")
    pipeline = SecurityLogAnalysisPipeline(aws_region='us-east-1')
    print("✓ Pipeline initialized")
    print()
    
    # Generate sample logs
    print("Generating sample security logs...")
    raw_logs = generate_sample_logs()
    print(f"✓ Generated {len(raw_logs)} raw log entries")
    print()
    
    # Ingest and parse logs
    print("=" * 80)
    print("LOG INGESTION AND PARSING")
    print("=" * 80)
    print()
    
    parsed_logs = pipeline.ingest_logs(raw_logs)
    print(f"Parsed {len(parsed_logs)} log entries")
    
    # Show parsing examples
    print("\nParsing Examples:")
    for log in parsed_logs[:3]:
        print(f"\n  Source: {log.source} ({log.source_type.value})")
        print(f"  Event Type: {log.event_type}")
        print(f"  Severity: {log.severity.value}")
        print(f"  Entities: {log.entities}")
        print(f"  Template: {log.template[:80]}...")
    
    # Log statistics
    print("\n" + "=" * 80)
    print("LOG ANALYSIS STATISTICS")
    print("=" * 80)
    print()
    
    stats = pipeline.get_analysis_statistics()
    print(f"Total Logs: {stats['total_logs']}")
    print(f"Unique Entities: {stats['unique_entities']}")
    
    print("\nBy Source Type:")
    for source, count in stats['by_source'].items():
        print(f"  {source}: {count}")
    
    print("\nBy Severity:")
    for severity, count in stats['by_severity'].items():
        print(f"  {severity}: {count}")
    
    # Run threat hunts
    print("\n" + "=" * 80)
    print("THREAT HUNTING RESULTS")
    print("=" * 80)
    print()
    
    hunt_results = pipeline.run_threat_hunts()
    
    for result in hunt_results:
        if result.matching_events:
            risk_indicator = "🔴" if result.risk_score > 0.7 else "🟠" if result.risk_score > 0.3 else "🟡"
            print(f"\n{risk_indicator} {result.query.name}")
            print(f"   MITRE: {result.query.mitre_technique}")
            print(f"   Matches: {len(result.matching_events)}")
            print(f"   Risk Score: {result.risk_score:.2f}")
            print(f"\n   Analysis:")
            print(f"   {result.ai_analysis}")
    
    # Create investigation
    print("\n" + "=" * 80)
    print("SECURITY INVESTIGATION")
    print("=" * 80)
    print()
    
    print("Creating investigation for suspicious user activity...")
    investigation = pipeline.investigate_entity(
        entity_type='username',
        entity_value='jsmith',
        hours_back=24
    )
    
    print(f"\nInvestigation: {investigation.title}")
    print(f"Events Found: {len(investigation.events)}")
    print(f"Evidence Hashes: {len(investigation.evidence_hashes)}")
    
    if investigation.events:
        print("\nTimeline:")
        for event in investigation.events[:5]:
            print(f"  [{event.timestamp}] {event.event_type}: {event.raw_message[:60]}...")
    
    # Generate investigation report
    print("\n" + "-" * 40)
    print("INVESTIGATION REPORT")
    print("-" * 40)
    
    report = pipeline.workbench.generate_investigation_report(investigation.investigation_id)
    print(report)
    
    # Show template analysis
    print("\n" + "=" * 80)
    print("LOG TEMPLATE ANALYSIS")
    print("=" * 80)
    print()
    
    rare_templates = pipeline.parser.template_extractor.get_rare_templates(threshold=1)
    print(f"Rare Templates (potential anomalies): {len(rare_templates)}")
    for template in rare_templates[:3]:
        print(f"  - {template[:80]}...")
    
    print("\n" + "=" * 80)
    print("Security Log Analysis Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
