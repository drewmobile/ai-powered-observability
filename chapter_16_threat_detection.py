"""
Chapter 16: Threat Detection with AI
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready AI-powered threat detection including:
- Network anomaly detection (data exfiltration, beaconing, unusual connections)
- Authentication anomaly detection (impossible travel, credential stuffing)
- Endpoint threat detection (suspicious processes, persistence mechanisms)
- Threat intelligence correlation
- Multi-signal attack correlation
- Kill chain detection across multiple stages
- AI-powered threat analysis with AWS Bedrock

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
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AttackStage(Enum):
    """Kill chain attack stages"""
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    COMMAND_AND_CONTROL = "command_and_control"


@dataclass
class NetworkEvent:
    """Represents a network flow event"""
    timestamp: datetime
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    bytes_sent: int
    bytes_received: int
    duration_seconds: float
    country_code: str
    is_internal: bool


@dataclass
class AuthenticationEvent:
    """Represents an authentication event"""
    timestamp: datetime
    username: str
    source_ip: str
    destination_host: str
    success: bool
    auth_method: str
    user_agent: str
    country_code: str
    mfa_used: bool


@dataclass
class EndpointEvent:
    """Represents an endpoint telemetry event"""
    timestamp: datetime
    hostname: str
    process_name: str
    parent_process: str
    command_line: str
    file_path: str
    registry_key: Optional[str]
    network_connections: List[str]
    user: str


@dataclass
class ThreatIndicator:
    """Represents a threat intelligence indicator"""
    indicator_type: str  # ip, domain, hash, email
    value: str
    threat_type: str
    confidence: float
    source: str
    last_seen: datetime


@dataclass
class ThreatAlert:
    """Represents a detected threat"""
    alert_id: str
    timestamp: datetime
    threat_type: str
    severity: ThreatSeverity
    attack_stage: Optional[AttackStage]
    source: str
    target: str
    description: str
    evidence: Dict[str, Any]
    indicators: List[str]
    recommended_actions: List[str]
    confidence: float
    mitre_techniques: List[str]


class NetworkAnomalyDetector:
    """
    Detects network-based threats including exfiltration, beaconing, and unusual connections.
    """
    
    def __init__(self, bedrock_client):
        """Initialize network anomaly detector"""
        self.bedrock_client = bedrock_client
        self.baseline_traffic: Dict[str, Dict[str, float]] = defaultdict(lambda: {'bytes_avg': 0, 'count': 0})
        self.known_internal_ips: Set[str] = set()
        self.suspicious_ports = {4444, 5555, 6666, 8888, 31337}  # Common malware ports
        self.suspicious_countries = {'CN', 'RU', 'KP', 'IR'}  # Example list
    
    def add_baseline_traffic(self, events: List[NetworkEvent]):
        """Build baseline of normal network traffic"""
        for event in events:
            key = event.source_ip
            self.baseline_traffic[key]['bytes_avg'] = (
                (self.baseline_traffic[key]['bytes_avg'] * self.baseline_traffic[key]['count'] + event.bytes_sent)
                / (self.baseline_traffic[key]['count'] + 1)
            )
            self.baseline_traffic[key]['count'] += 1
            
            if event.is_internal:
                self.known_internal_ips.add(event.source_ip)
                self.known_internal_ips.add(event.destination_ip)
    
    def detect_threats(self, events: List[NetworkEvent]) -> List[ThreatAlert]:
        """
        Detect network-based threats.
        
        Args:
            events: Network events to analyze
            
        Returns:
            List of detected threats
        """
        alerts = []
        
        for event in events:
            # Check for data exfiltration
            exfil_alert = self._detect_data_exfiltration(event)
            if exfil_alert:
                alerts.append(exfil_alert)
            
            # Check for beaconing
            beaconing_alert = self._detect_beaconing(events, event.source_ip)
            if beaconing_alert and beaconing_alert.alert_id not in [a.alert_id for a in alerts]:
                alerts.append(beaconing_alert)
            
            # Check for suspicious connections
            suspicious_alert = self._detect_suspicious_connection(event)
            if suspicious_alert:
                alerts.append(suspicious_alert)
        
        return alerts
    
    def _detect_data_exfiltration(self, event: NetworkEvent) -> Optional[ThreatAlert]:
        """Detect potential data exfiltration"""
        baseline = self.baseline_traffic.get(event.source_ip, {})
        avg_bytes = baseline.get('bytes_avg', 0)
        
        # Flag if transfer is 10x normal or above threshold
        if event.bytes_sent > 100_000_000 or (avg_bytes > 0 and event.bytes_sent > avg_bytes * 10):
            return ThreatAlert(
                alert_id=f"exfil_{event.source_ip}_{event.timestamp.timestamp()}",
                timestamp=event.timestamp,
                threat_type="data_exfiltration",
                severity=ThreatSeverity.HIGH,
                attack_stage=AttackStage.EXFILTRATION,
                source=event.source_ip,
                target=event.destination_ip,
                description=f"Large data transfer ({event.bytes_sent / 1_000_000:.1f}MB) from {event.source_ip} to external IP {event.destination_ip}",
                evidence={
                    'bytes_sent': event.bytes_sent,
                    'baseline_avg': avg_bytes,
                    'destination_country': event.country_code,
                    'duration_seconds': event.duration_seconds
                },
                indicators=[event.destination_ip],
                recommended_actions=[
                    "Isolate source host immediately",
                    "Block destination IP",
                    "Investigate data accessed on source host",
                    "Check for compromised credentials"
                ],
                confidence=0.85,
                mitre_techniques=["T1048 - Exfiltration Over Alternative Protocol"]
            )
        return None
    
    def _detect_beaconing(self, events: List[NetworkEvent], source_ip: str) -> Optional[ThreatAlert]:
        """Detect C2 beaconing patterns"""
        # Filter events from this source
        source_events = [e for e in events if e.source_ip == source_ip]
        
        if len(source_events) < 5:
            return None
        
        # Check for regular intervals (beaconing)
        timestamps = sorted([e.timestamp for e in source_events])
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        
        if not intervals:
            return None
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Low variance in intervals suggests beaconing
        if std_interval < avg_interval * 0.1 and len(intervals) >= 5:
            destination = source_events[0].destination_ip
            
            return ThreatAlert(
                alert_id=f"beacon_{source_ip}_{destination}",
                timestamp=datetime.now(),
                threat_type="c2_beaconing",
                severity=ThreatSeverity.HIGH,
                attack_stage=AttackStage.COMMAND_AND_CONTROL,
                source=source_ip,
                target=destination,
                description=f"Regular beaconing pattern detected from {source_ip} to {destination} (interval: {avg_interval:.1f}s ± {std_interval:.1f}s)",
                evidence={
                    'average_interval_seconds': avg_interval,
                    'interval_stddev': std_interval,
                    'beacon_count': len(intervals) + 1,
                    'destination_ip': destination
                },
                indicators=[destination],
                recommended_actions=[
                    "Block outbound connection to destination",
                    "Isolate affected host",
                    "Analyze traffic content for C2 protocol",
                    "Search for related malware on host"
                ],
                confidence=0.9,
                mitre_techniques=["T1071 - Application Layer Protocol", "T1573 - Encrypted Channel"]
            )
        return None
    
    def _detect_suspicious_connection(self, event: NetworkEvent) -> Optional[ThreatAlert]:
        """Detect suspicious network connections"""
        # Check for connections to suspicious ports
        if event.destination_port in self.suspicious_ports:
            return ThreatAlert(
                alert_id=f"susport_{event.source_ip}_{event.destination_port}_{event.timestamp.timestamp()}",
                timestamp=event.timestamp,
                threat_type="suspicious_port",
                severity=ThreatSeverity.MEDIUM,
                attack_stage=AttackStage.COMMAND_AND_CONTROL,
                source=event.source_ip,
                target=f"{event.destination_ip}:{event.destination_port}",
                description=f"Connection to suspicious port {event.destination_port}",
                evidence={
                    'destination_port': event.destination_port,
                    'destination_ip': event.destination_ip
                },
                indicators=[event.destination_ip],
                recommended_actions=[
                    "Investigate process making connection",
                    "Block port if not legitimate"
                ],
                confidence=0.7,
                mitre_techniques=["T1571 - Non-Standard Port"]
            )
        
        # Check for connections to suspicious countries
        if event.country_code in self.suspicious_countries and not event.is_internal:
            return ThreatAlert(
                alert_id=f"susgeo_{event.source_ip}_{event.country_code}_{event.timestamp.timestamp()}",
                timestamp=event.timestamp,
                threat_type="suspicious_geo",
                severity=ThreatSeverity.MEDIUM,
                attack_stage=None,
                source=event.source_ip,
                target=event.destination_ip,
                description=f"Connection to high-risk geography ({event.country_code})",
                evidence={
                    'country_code': event.country_code,
                    'destination_ip': event.destination_ip,
                    'bytes_sent': event.bytes_sent
                },
                indicators=[event.destination_ip],
                recommended_actions=[
                    "Verify if connection is legitimate business traffic",
                    "Review user's normal access patterns"
                ],
                confidence=0.5,
                mitre_techniques=[]
            )
        
        return None


class AuthenticationAnomalyDetector:
    """
    Detects authentication-based threats including credential stuffing,
    impossible travel, and account takeover.
    """
    
    def __init__(self, bedrock_client):
        """Initialize authentication detector"""
        self.bedrock_client = bedrock_client
        self.user_profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'known_ips': set(),
            'known_countries': set(),
            'typical_hours': set(),
            'accessed_hosts': set()
        })
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
    
    def build_user_profiles(self, events: List[AuthenticationEvent]):
        """Build baseline user authentication profiles"""
        for event in events:
            if event.success:
                profile = self.user_profiles[event.username]
                profile['known_ips'].add(event.source_ip)
                profile['known_countries'].add(event.country_code)
                profile['typical_hours'].add(event.timestamp.hour)
                profile['accessed_hosts'].add(event.destination_host)
    
    def detect_threats(self, events: List[AuthenticationEvent]) -> List[ThreatAlert]:
        """
        Detect authentication-based threats.
        
        Args:
            events: Authentication events to analyze
            
        Returns:
            List of detected threats
        """
        alerts = []
        
        for event in events:
            # Check for impossible travel
            travel_alert = self._detect_impossible_travel(event)
            if travel_alert:
                alerts.append(travel_alert)
            
            # Check for credential stuffing
            stuffing_alert = self._detect_credential_stuffing(events)
            if stuffing_alert and stuffing_alert.alert_id not in [a.alert_id for a in alerts]:
                alerts.append(stuffing_alert)
            
            # Check for account takeover indicators
            takeover_alert = self._detect_account_takeover(event)
            if takeover_alert:
                alerts.append(takeover_alert)
        
        return alerts
    
    def _detect_impossible_travel(self, event: AuthenticationEvent) -> Optional[ThreatAlert]:
        """Detect logins from impossible geographic locations"""
        profile = self.user_profiles.get(event.username)
        
        if not profile or not profile['known_countries']:
            return None
        
        # Check if login from new country
        if event.country_code not in profile['known_countries']:
            # In production, would calculate actual geographic distance and time
            return ThreatAlert(
                alert_id=f"travel_{event.username}_{event.timestamp.timestamp()}",
                timestamp=event.timestamp,
                threat_type="impossible_travel",
                severity=ThreatSeverity.HIGH,
                attack_stage=AttackStage.INITIAL_ACCESS,
                source=event.source_ip,
                target=event.username,
                description=f"User {event.username} logged in from {event.country_code}, never seen from this country before",
                evidence={
                    'username': event.username,
                    'new_country': event.country_code,
                    'known_countries': list(profile['known_countries']),
                    'source_ip': event.source_ip,
                    'mfa_used': event.mfa_used
                },
                indicators=[event.source_ip],
                recommended_actions=[
                    "Force password reset",
                    "Require MFA verification",
                    "Review recent account activity",
                    "Contact user to verify legitimacy"
                ],
                confidence=0.85 if not event.mfa_used else 0.6,
                mitre_techniques=["T1078 - Valid Accounts"]
            )
        return None
    
    def _detect_credential_stuffing(self, events: List[AuthenticationEvent]) -> Optional[ThreatAlert]:
        """Detect credential stuffing attacks"""
        # Group by source IP
        failures_by_ip = defaultdict(list)
        
        for event in events:
            if not event.success:
                failures_by_ip[event.source_ip].append(event)
        
        # Check for IPs with many failed attempts to different users
        for ip, failures in failures_by_ip.items():
            unique_users = set(e.username for e in failures)
            
            if len(unique_users) >= 10 and len(failures) >= 20:
                return ThreatAlert(
                    alert_id=f"stuffing_{ip}",
                    timestamp=datetime.now(),
                    threat_type="credential_stuffing",
                    severity=ThreatSeverity.CRITICAL,
                    attack_stage=AttackStage.CREDENTIAL_ACCESS,
                    source=ip,
                    target="multiple_accounts",
                    description=f"Credential stuffing attack from {ip}: {len(failures)} failed attempts against {len(unique_users)} unique accounts",
                    evidence={
                        'source_ip': ip,
                        'failed_attempts': len(failures),
                        'unique_users': len(unique_users),
                        'sample_users': list(unique_users)[:5]
                    },
                    indicators=[ip],
                    recommended_actions=[
                        "Block source IP immediately",
                        "Enable account lockout policies",
                        "Force password reset for targeted accounts",
                        "Review successful logins from this IP"
                    ],
                    confidence=0.95,
                    mitre_techniques=["T1110 - Brute Force", "T1110.004 - Credential Stuffing"]
                )
        return None
    
    def _detect_account_takeover(self, event: AuthenticationEvent) -> Optional[ThreatAlert]:
        """Detect potential account takeover"""
        profile = self.user_profiles.get(event.username)
        
        if not profile or not event.success:
            return None
        
        anomaly_score = 0
        anomalies = []
        
        # Check for new IP
        if event.source_ip not in profile['known_ips']:
            anomaly_score += 1
            anomalies.append(f"New IP: {event.source_ip}")
        
        # Check for unusual hour
        if event.timestamp.hour not in profile['typical_hours']:
            anomaly_score += 1
            anomalies.append(f"Unusual hour: {event.timestamp.hour}")
        
        # Check for new country
        if event.country_code not in profile['known_countries']:
            anomaly_score += 2
            anomalies.append(f"New country: {event.country_code}")
        
        if anomaly_score >= 2:
            return ThreatAlert(
                alert_id=f"takeover_{event.username}_{event.timestamp.timestamp()}",
                timestamp=event.timestamp,
                threat_type="account_takeover",
                severity=ThreatSeverity.HIGH if anomaly_score >= 3 else ThreatSeverity.MEDIUM,
                attack_stage=AttackStage.INITIAL_ACCESS,
                source=event.source_ip,
                target=event.username,
                description=f"Potential account takeover for {event.username}: {', '.join(anomalies)}",
                evidence={
                    'username': event.username,
                    'anomalies': anomalies,
                    'anomaly_score': anomaly_score,
                    'source_ip': event.source_ip
                },
                indicators=[event.source_ip],
                recommended_actions=[
                    "Monitor account activity closely",
                    "Consider forcing reauthentication",
                    "Enable MFA if not already enabled"
                ],
                confidence=min(0.5 + anomaly_score * 0.15, 0.95),
                mitre_techniques=["T1078 - Valid Accounts"]
            )
        return None


class EndpointThreatDetector:
    """
    Detects endpoint-based threats including malware execution,
    persistence mechanisms, and suspicious processes.
    """
    
    def __init__(self, bedrock_client):
        """Initialize endpoint threat detector"""
        self.bedrock_client = bedrock_client
        self.suspicious_parent_child = [
            ('excel.exe', 'powershell.exe'),
            ('excel.exe', 'cmd.exe'),
            ('word.exe', 'powershell.exe'),
            ('outlook.exe', 'powershell.exe'),
            ('svchost.exe', 'cmd.exe'),
        ]
        self.suspicious_processes = {'mimikatz', 'psexec', 'procdump', 'lazagne'}
        self.persistence_registry_keys = [
            'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run',
            'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run',
            'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnce'
        ]
    
    def detect_threats(self, events: List[EndpointEvent]) -> List[ThreatAlert]:
        """
        Detect endpoint-based threats.
        
        Args:
            events: Endpoint events to analyze
            
        Returns:
            List of detected threats
        """
        alerts = []
        
        for event in events:
            # Check for suspicious parent-child relationships
            proc_alert = self._detect_suspicious_process(event)
            if proc_alert:
                alerts.append(proc_alert)
            
            # Check for persistence mechanisms
            persist_alert = self._detect_persistence(event)
            if persist_alert:
                alerts.append(persist_alert)
            
            # Check for encoded commands
            encoded_alert = self._detect_encoded_commands(event)
            if encoded_alert:
                alerts.append(encoded_alert)
        
        return alerts
    
    def _detect_suspicious_process(self, event: EndpointEvent) -> Optional[ThreatAlert]:
        """Detect suspicious process execution"""
        # Check parent-child relationship
        parent = event.parent_process.lower()
        child = event.process_name.lower()
        
        for sus_parent, sus_child in self.suspicious_parent_child:
            if sus_parent in parent and sus_child in child:
                return ThreatAlert(
                    alert_id=f"susprocess_{event.hostname}_{event.timestamp.timestamp()}",
                    timestamp=event.timestamp,
                    threat_type="suspicious_process",
                    severity=ThreatSeverity.HIGH,
                    attack_stage=AttackStage.EXECUTION,
                    source=event.hostname,
                    target=event.process_name,
                    description=f"Suspicious process tree: {event.parent_process} spawned {event.process_name}",
                    evidence={
                        'hostname': event.hostname,
                        'process': event.process_name,
                        'parent': event.parent_process,
                        'command_line': event.command_line,
                        'user': event.user
                    },
                    indicators=[event.process_name],
                    recommended_actions=[
                        "Isolate the host",
                        "Analyze command line arguments",
                        "Check for lateral movement from this host",
                        "Review recent file downloads"
                    ],
                    confidence=0.9,
                    mitre_techniques=["T1059 - Command and Scripting Interpreter"]
                )
        
        # Check for known malicious tools
        if any(sus in child for sus in self.suspicious_processes):
            return ThreatAlert(
                alert_id=f"maltool_{event.hostname}_{event.timestamp.timestamp()}",
                timestamp=event.timestamp,
                threat_type="malicious_tool",
                severity=ThreatSeverity.CRITICAL,
                attack_stage=AttackStage.CREDENTIAL_ACCESS,
                source=event.hostname,
                target=event.process_name,
                description=f"Known malicious tool executed: {event.process_name}",
                evidence={
                    'hostname': event.hostname,
                    'process': event.process_name,
                    'command_line': event.command_line,
                    'user': event.user
                },
                indicators=[event.process_name],
                recommended_actions=[
                    "Isolate host immediately",
                    "Assume credential compromise",
                    "Reset all credentials from this host",
                    "Full forensic investigation required"
                ],
                confidence=0.98,
                mitre_techniques=["T1003 - OS Credential Dumping"]
            )
        
        return None
    
    def _detect_persistence(self, event: EndpointEvent) -> Optional[ThreatAlert]:
        """Detect persistence mechanism creation"""
        if event.registry_key:
            for persist_key in self.persistence_registry_keys:
                if persist_key.lower() in event.registry_key.lower():
                    return ThreatAlert(
                        alert_id=f"persist_{event.hostname}_{event.timestamp.timestamp()}",
                        timestamp=event.timestamp,
                        threat_type="persistence_mechanism",
                        severity=ThreatSeverity.HIGH,
                        attack_stage=AttackStage.PERSISTENCE,
                        source=event.hostname,
                        target=event.registry_key,
                        description=f"Persistence mechanism detected: Registry key modified for auto-run",
                        evidence={
                            'hostname': event.hostname,
                            'registry_key': event.registry_key,
                            'process': event.process_name,
                            'file_path': event.file_path
                        },
                        indicators=[event.file_path],
                        recommended_actions=[
                            "Analyze the auto-run entry",
                            "Check file reputation",
                            "Remove if malicious",
                            "Investigate initial infection vector"
                        ],
                        confidence=0.8,
                        mitre_techniques=["T1547.001 - Registry Run Keys"]
                    )
        return None
    
    def _detect_encoded_commands(self, event: EndpointEvent) -> Optional[ThreatAlert]:
        """Detect encoded/obfuscated command execution"""
        cmd_lower = event.command_line.lower()
        
        # Check for base64 encoded PowerShell
        if 'powershell' in cmd_lower and ('-enc' in cmd_lower or '-encoded' in cmd_lower or 'frombase64' in cmd_lower):
            return ThreatAlert(
                alert_id=f"encoded_{event.hostname}_{event.timestamp.timestamp()}",
                timestamp=event.timestamp,
                threat_type="encoded_command",
                severity=ThreatSeverity.HIGH,
                attack_stage=AttackStage.EXECUTION,
                source=event.hostname,
                target="powershell.exe",
                description="Encoded PowerShell command detected",
                evidence={
                    'hostname': event.hostname,
                    'command_line': event.command_line[:500],  # Truncate for display
                    'parent_process': event.parent_process,
                    'user': event.user
                },
                indicators=[],
                recommended_actions=[
                    "Decode and analyze the command",
                    "Check for network connections",
                    "Isolate if malicious",
                    "Review how user obtained the command"
                ],
                confidence=0.85,
                mitre_techniques=["T1027 - Obfuscated Files or Information", "T1059.001 - PowerShell"]
            )
        return None


class ThreatIntelligenceCorrelator:
    """
    Correlates events against threat intelligence indicators.
    """
    
    def __init__(self, bedrock_client):
        """Initialize threat intelligence correlator"""
        self.bedrock_client = bedrock_client
        self.indicators: Dict[str, ThreatIndicator] = {}
    
    def add_indicators(self, indicators: List[ThreatIndicator]):
        """Add threat intelligence indicators"""
        for indicator in indicators:
            self.indicators[indicator.value.lower()] = indicator
    
    def correlate(self,
                 network_events: List[NetworkEvent],
                 auth_events: List[AuthenticationEvent]) -> List[ThreatAlert]:
        """
        Correlate events against threat intelligence.
        
        Returns:
            List of threat intelligence matches
        """
        alerts = []
        
        # Check network events
        for event in network_events:
            if event.destination_ip.lower() in self.indicators:
                indicator = self.indicators[event.destination_ip.lower()]
                alerts.append(self._create_ti_alert(event.source_ip, indicator, event.timestamp))
        
        # Check auth events
        for event in auth_events:
            if event.source_ip.lower() in self.indicators:
                indicator = self.indicators[event.source_ip.lower()]
                alerts.append(self._create_ti_alert(event.username, indicator, event.timestamp))
        
        return alerts
    
    def _create_ti_alert(self,
                        source: str,
                        indicator: ThreatIndicator,
                        timestamp: datetime) -> ThreatAlert:
        """Create alert from threat intelligence match"""
        return ThreatAlert(
            alert_id=f"ti_{indicator.value}_{timestamp.timestamp()}",
            timestamp=timestamp,
            threat_type="threat_intelligence_match",
            severity=ThreatSeverity.HIGH if indicator.confidence > 0.8 else ThreatSeverity.MEDIUM,
            attack_stage=None,
            source=source,
            target=indicator.value,
            description=f"Connection to known malicious indicator: {indicator.value} ({indicator.threat_type})",
            evidence={
                'indicator': indicator.value,
                'indicator_type': indicator.indicator_type,
                'threat_type': indicator.threat_type,
                'confidence': indicator.confidence,
                'source': indicator.source
            },
            indicators=[indicator.value],
            recommended_actions=[
                "Block indicator immediately",
                "Investigate source for compromise",
                "Review historical connections to this indicator"
            ],
            confidence=indicator.confidence,
            mitre_techniques=[]
        )


class KillChainDetector:
    """
    Detects multi-stage attacks by correlating alerts across kill chain stages.
    """
    
    def __init__(self, bedrock_client):
        """Initialize kill chain detector"""
        self.bedrock_client = bedrock_client
        self.alerts_by_entity: Dict[str, List[ThreatAlert]] = defaultdict(list)
    
    def add_alert(self, alert: ThreatAlert):
        """Add alert for kill chain correlation"""
        # Index by source and target
        self.alerts_by_entity[alert.source].append(alert)
        self.alerts_by_entity[alert.target].append(alert)
    
    def detect_kill_chain(self) -> List[ThreatAlert]:
        """
        Detect complete or partial kill chain attacks.
        
        Returns:
            List of kill chain detection alerts
        """
        kill_chain_alerts = []
        
        for entity, alerts in self.alerts_by_entity.items():
            if len(alerts) < 2:
                continue
            
            # Get unique attack stages
            stages = set(a.attack_stage for a in alerts if a.attack_stage)
            
            # Check for multi-stage attack
            if len(stages) >= 3:
                # Calculate kill chain coverage
                all_stages = list(AttackStage)
                coverage = len(stages) / len(all_stages)
                
                kill_chain_alert = ThreatAlert(
                    alert_id=f"killchain_{entity}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    threat_type="kill_chain_detected",
                    severity=ThreatSeverity.CRITICAL,
                    attack_stage=None,
                    source=entity,
                    target="multiple",
                    description=f"Multi-stage attack detected involving {entity}. {len(stages)} attack stages identified.",
                    evidence={
                        'entity': entity,
                        'stages_detected': [s.value for s in stages],
                        'alert_count': len(alerts),
                        'kill_chain_coverage': coverage,
                        'related_alerts': [a.alert_id for a in alerts]
                    },
                    indicators=list(set(i for a in alerts for i in a.indicators)),
                    recommended_actions=[
                        "Initiate incident response immediately",
                        "Isolate all affected systems",
                        "Preserve evidence for forensics",
                        "Reset all potentially compromised credentials",
                        "Notify security leadership"
                    ],
                    confidence=min(0.5 + coverage, 0.99),
                    mitre_techniques=list(set(t for a in alerts for t in a.mitre_techniques))
                )
                
                kill_chain_alerts.append(kill_chain_alert)
        
        return kill_chain_alerts


class ThreatDetectionPipeline:
    """
    Production pipeline orchestrating all threat detection components.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize threat detection pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.network_detector = NetworkAnomalyDetector(self.bedrock_client)
        self.auth_detector = AuthenticationAnomalyDetector(self.bedrock_client)
        self.endpoint_detector = EndpointThreatDetector(self.bedrock_client)
        self.ti_correlator = ThreatIntelligenceCorrelator(self.bedrock_client)
        self.kill_chain_detector = KillChainDetector(self.bedrock_client)
        
        self.all_alerts: List[ThreatAlert] = []
    
    def load_threat_intelligence(self, indicators: List[ThreatIndicator]):
        """Load threat intelligence indicators"""
        self.ti_correlator.add_indicators(indicators)
    
    def process_events(self,
                      network_events: List[NetworkEvent] = None,
                      auth_events: List[AuthenticationEvent] = None,
                      endpoint_events: List[EndpointEvent] = None) -> List[ThreatAlert]:
        """
        Process all event types and detect threats.
        
        Args:
            network_events: Network flow events
            auth_events: Authentication events
            endpoint_events: Endpoint telemetry events
            
        Returns:
            All detected threats
        """
        alerts = []
        
        # Network threat detection
        if network_events:
            network_alerts = self.network_detector.detect_threats(network_events)
            alerts.extend(network_alerts)
        
        # Authentication threat detection
        if auth_events:
            auth_alerts = self.auth_detector.detect_threats(auth_events)
            alerts.extend(auth_alerts)
        
        # Endpoint threat detection
        if endpoint_events:
            endpoint_alerts = self.endpoint_detector.detect_threats(endpoint_events)
            alerts.extend(endpoint_alerts)
        
        # Threat intelligence correlation
        if network_events or auth_events:
            ti_alerts = self.ti_correlator.correlate(
                network_events or [],
                auth_events or []
            )
            alerts.extend(ti_alerts)
        
        # Add alerts to kill chain detector
        for alert in alerts:
            self.kill_chain_detector.add_alert(alert)
        
        # Check for kill chain patterns
        kill_chain_alerts = self.kill_chain_detector.detect_kill_chain()
        alerts.extend(kill_chain_alerts)
        
        self.all_alerts.extend(alerts)
        
        return alerts
    
    def generate_threat_summary(self, alerts: List[ThreatAlert]) -> str:
        """Generate AI-powered threat summary"""
        if not alerts:
            return "No threats detected."
        
        alerts_text = "\n".join([
            f"- {a.threat_type}: {a.description} (Severity: {a.severity.value})"
            for a in alerts[:10]
        ])
        
        prompt = f"""Analyze these security alerts and provide an executive summary:

Alerts Detected:
{alerts_text}

Total Alerts: {len(alerts)}

Provide a brief (3-4 sentences) executive summary covering:
1. Overall threat level
2. Most critical concerns
3. Immediate actions recommended

Be concise and actionable."""

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
            return f"Could not generate summary: {e}"
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        if not self.all_alerts:
            return {'no_threats': True}
        
        by_severity = Counter(a.severity.value for a in self.all_alerts)
        by_type = Counter(a.threat_type for a in self.all_alerts)
        by_stage = Counter(a.attack_stage.value for a in self.all_alerts if a.attack_stage)
        
        return {
            'total_alerts': len(self.all_alerts),
            'by_severity': dict(by_severity),
            'by_type': dict(by_type),
            'by_attack_stage': dict(by_stage),
            'unique_sources': len(set(a.source for a in self.all_alerts)),
            'unique_indicators': len(set(i for a in self.all_alerts for i in a.indicators))
        }


# Example usage and realistic scenarios
def generate_sample_attack_scenario() -> Tuple[List[NetworkEvent], List[AuthenticationEvent], List[EndpointEvent]]:
    """Generate a realistic multi-stage attack scenario"""
    
    base_time = datetime.now() - timedelta(hours=4)
    
    # Network events including exfiltration and C2
    network_events = [
        # Normal traffic
        NetworkEvent(base_time, "192.168.1.50", "8.8.8.8", 54321, 443, "TCP", 1000, 5000, 0.5, "US", False),
        # C2 beaconing
        NetworkEvent(base_time + timedelta(minutes=10), "192.168.1.100", "185.143.223.47", 49152, 443, "TCP", 500, 200, 0.1, "RU", False),
        NetworkEvent(base_time + timedelta(minutes=20), "192.168.1.100", "185.143.223.47", 49153, 443, "TCP", 500, 200, 0.1, "RU", False),
        NetworkEvent(base_time + timedelta(minutes=30), "192.168.1.100", "185.143.223.47", 49154, 443, "TCP", 500, 200, 0.1, "RU", False),
        NetworkEvent(base_time + timedelta(minutes=40), "192.168.1.100", "185.143.223.47", 49155, 443, "TCP", 500, 200, 0.1, "RU", False),
        NetworkEvent(base_time + timedelta(minutes=50), "192.168.1.100", "185.143.223.47", 49156, 443, "TCP", 500, 200, 0.1, "RU", False),
        # Data exfiltration
        NetworkEvent(base_time + timedelta(hours=3), "192.168.1.100", "103.25.43.12", 52000, 443, "TCP", 150_000_000, 1000, 3600, "CN", False),
    ]
    
    # Authentication events including compromise
    auth_events = [
        # Normal login
        AuthenticationEvent(base_time, "jsmith", "192.168.1.50", "fileserver01", True, "password", "Windows", "US", True),
        # Suspicious login (new country)
        AuthenticationEvent(base_time + timedelta(hours=1), "jsmith", "185.143.223.100", "dc01", True, "password", "curl/7.68", "RU", False),
        # Lateral movement
        AuthenticationEvent(base_time + timedelta(hours=2), "jsmith", "192.168.1.100", "db-server", True, "ntlm", "Windows", "US", False),
        AuthenticationEvent(base_time + timedelta(hours=2, minutes=5), "jsmith", "192.168.1.100", "backup-server", True, "ntlm", "Windows", "US", False),
        # Credential stuffing from attacker
        *[AuthenticationEvent(base_time + timedelta(minutes=i), f"user{i}", "45.33.32.156", "vpn01", False, "password", "Python/3.8", "NL", False) for i in range(25)]
    ]
    
    # Endpoint events including malware execution
    endpoint_events = [
        # Malicious macro execution
        EndpointEvent(
            base_time + timedelta(hours=1),
            "WORKSTATION-100",
            "powershell.exe",
            "excel.exe",
            "powershell.exe -encodedcommand SQBFAFgAIAAoAE4AZQB3...",
            "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
            None,
            ["185.143.223.47:443"],
            "jsmith"
        ),
        # Persistence mechanism
        EndpointEvent(
            base_time + timedelta(hours=1, minutes=5),
            "WORKSTATION-100",
            "reg.exe",
            "powershell.exe",
            "reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run /v Updater /t REG_SZ /d C:\\Users\\jsmith\\AppData\\update.exe",
            "C:\\Windows\\System32\\reg.exe",
            "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
            [],
            "jsmith"
        ),
        # Credential dumping
        EndpointEvent(
            base_time + timedelta(hours=2),
            "WORKSTATION-100",
            "mimikatz.exe",
            "cmd.exe",
            "mimikatz.exe privilege::debug sekurlsa::logonpasswords exit",
            "C:\\temp\\mimikatz.exe",
            None,
            [],
            "jsmith"
        )
    ]
    
    return network_events, auth_events, endpoint_events


def main():
    """
    Demonstrate threat detection pipeline with realistic attack scenario.
    """
    print("=" * 80)
    print("Chapter 16: Threat Detection with AI")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing threat detection pipeline...")
    pipeline = ThreatDetectionPipeline(aws_region='us-east-1')
    
    # Load threat intelligence
    ti_indicators = [
        ThreatIndicator("ip", "185.143.223.47", "C2 Server", 0.95, "ThreatFeed", datetime.now()),
        ThreatIndicator("ip", "103.25.43.12", "Exfiltration Endpoint", 0.9, "ThreatFeed", datetime.now()),
        ThreatIndicator("ip", "45.33.32.156", "Known Attacker", 0.85, "ThreatFeed", datetime.now()),
    ]
    pipeline.load_threat_intelligence(ti_indicators)
    print(f"✓ Loaded {len(ti_indicators)} threat intelligence indicators")
    print()
    
    # Generate attack scenario
    print("Simulating multi-stage attack scenario...")
    network_events, auth_events, endpoint_events = generate_sample_attack_scenario()
    print(f"  - {len(network_events)} network events")
    print(f"  - {len(auth_events)} authentication events")
    print(f"  - {len(endpoint_events)} endpoint events")
    print()
    
    # Build baselines
    pipeline.network_detector.add_baseline_traffic(network_events[:2])
    pipeline.auth_detector.build_user_profiles(auth_events[:1])
    
    # Process events and detect threats
    print("=" * 80)
    print("THREAT DETECTION RESULTS")
    print("=" * 80)
    print()
    
    alerts = pipeline.process_events(
        network_events=network_events,
        auth_events=auth_events,
        endpoint_events=endpoint_events
    )
    
    # Display alerts by severity
    for severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH, ThreatSeverity.MEDIUM, ThreatSeverity.LOW]:
        severity_alerts = [a for a in alerts if a.severity == severity]
        
        if severity_alerts:
            print(f"\n{'🔴' if severity == ThreatSeverity.CRITICAL else '🟠' if severity == ThreatSeverity.HIGH else '🟡' if severity == ThreatSeverity.MEDIUM else '🟢'} {severity.value.upper()} ALERTS ({len(severity_alerts)})")
            print("-" * 60)
            
            for alert in severity_alerts[:5]:  # Show first 5 per severity
                print(f"\n  [{alert.threat_type}]")
                print(f"  {alert.description}")
                print(f"  Source: {alert.source} → Target: {alert.target}")
                if alert.attack_stage:
                    print(f"  Attack Stage: {alert.attack_stage.value}")
                print(f"  Confidence: {alert.confidence:.0%}")
                if alert.mitre_techniques:
                    print(f"  MITRE: {', '.join(alert.mitre_techniques[:2])}")
    
    # Generate executive summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()
    
    summary = pipeline.generate_threat_summary(alerts)
    print(summary)
    
    # Statistics
    print("\n" + "=" * 80)
    print("THREAT STATISTICS")
    print("=" * 80)
    print()
    
    stats = pipeline.get_threat_statistics()
    
    print(f"Total Alerts: {stats['total_alerts']}")
    print(f"Unique Sources: {stats['unique_sources']}")
    print(f"Unique Indicators: {stats['unique_indicators']}")
    
    print(f"\nBy Severity:")
    for severity, count in stats['by_severity'].items():
        print(f"  {severity}: {count}")
    
    print(f"\nBy Attack Stage:")
    for stage, count in stats['by_attack_stage'].items():
        print(f"  {stage}: {count}")
    
    print("\n" + "=" * 80)
    print("Threat Detection Demonstration Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
