"""
Chapter 19: Automated Security Incident Response
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready automated security incident response including:
- Security response playbooks (phishing, malware, compromised credentials, exfiltration, ransomware)
- Multi-level automation (assisted, human-approved, bounded, adaptive)
- Safety mechanisms (blast radius limits, rate limiting, kill switches)
- Evidence preservation and chain of custody
- Escalation management and human-in-the-loop
- Incident timeline and documentation
- AI-powered incident analysis with AWS Bedrock

Author: AI-Powered Observability
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Callable, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import warnings
warnings.filterwarnings('ignore')


class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(Enum):
    """Incident lifecycle status"""
    DETECTED = "detected"
    TRIAGING = "triaging"
    CONTAINING = "containing"
    ERADICATING = "eradicating"
    RECOVERING = "recovering"
    CLOSED = "closed"
    ESCALATED = "escalated"


class ActionStatus(Enum):
    """Response action status"""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    BLOCKED = "blocked"


class AutomationLevel(Enum):
    """Level of automation for response actions"""
    MANUAL = "manual"  # Human executes
    ASSISTED = "assisted"  # Automation helps, human decides
    HUMAN_APPROVED = "human_approved"  # Automation proposes, human approves
    BOUNDED = "bounded"  # Auto-execute within bounds
    ADAPTIVE = "adaptive"  # AI-driven response


class ActionCategory(Enum):
    """Categories of response actions"""
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    EVIDENCE = "evidence"
    NOTIFICATION = "notification"
    RECOVERY = "recovery"


@dataclass
class SecurityIncident:
    """Represents a security incident"""
    incident_id: str
    timestamp: datetime
    incident_type: str  # phishing, malware, credential_compromise, etc.
    severity: IncidentSeverity
    status: IncidentStatus
    source_alert_id: str
    affected_entities: List[str]
    indicators: List[str]
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    assigned_analyst: Optional[str] = None
    playbook_id: Optional[str] = None


@dataclass
class ResponseAction:
    """Represents a response action"""
    action_id: str
    incident_id: str
    timestamp: datetime
    action_type: str
    category: ActionCategory
    target: str
    parameters: Dict[str, Any]
    automation_level: AutomationLevel
    status: ActionStatus
    requires_approval: bool
    approved_by: Optional[str] = None
    executed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    rollback_action: Optional['ResponseAction'] = None
    error_message: Optional[str] = None


@dataclass
class PlaybookStep:
    """Represents a step in a response playbook"""
    step_id: str
    name: str
    description: str
    action_type: str
    category: ActionCategory
    automation_level: AutomationLevel
    condition: Optional[Callable] = None  # Function to evaluate if step should run
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    continue_on_failure: bool = False


@dataclass
class ResponsePlaybook:
    """Represents a security response playbook"""
    playbook_id: str
    name: str
    description: str
    trigger_types: List[str]  # Incident types that trigger this playbook
    severity_threshold: IncidentSeverity
    steps: List[PlaybookStep]
    escalation_timeout_minutes: int = 30
    enabled: bool = True


@dataclass
class EscalationPath:
    """Defines escalation routing"""
    level: int
    role: str
    contact_method: str  # email, page, slack
    timeout_minutes: int


class SafetyController:
    """
    Implements safety mechanisms to prevent automation from causing harm.
    """
    
    def __init__(self):
        """Initialize safety controller"""
        self.action_counts: Dict[str, List[datetime]] = defaultdict(list)
        self.affected_entities: Dict[str, Set[str]] = defaultdict(set)
        self.global_kill_switch = False
        self.playbook_kill_switches: Dict[str, bool] = {}
        self.scope_kill_switches: Dict[str, bool] = {}
        
        # Safety limits
        self.rate_limits = {
            'isolate_host': 10,  # Max per hour
            'disable_account': 20,
            'block_ip': 50,
            'quarantine_email': 100
        }
        
        self.blast_radius_limits = {
            'max_hosts_isolated': 5,
            'max_accounts_disabled': 10,
            'max_ips_blocked': 100
        }
        
        # Prohibited actions
        self.prohibited_actions = {'delete_data', 'destroy_system', 'modify_backup'}
        
        # Actions requiring approval
        self.approval_required = {
            'disable_privileged_account',
            'isolate_production_host',
            'block_wide_ip_range',
            'modify_firewall_rules'
        }
    
    def can_execute(self, action: ResponseAction) -> Tuple[bool, str]:
        """
        Check if action can be safely executed.
        
        Returns:
            Tuple of (can_execute, reason)
        """
        # Check global kill switch
        if self.global_kill_switch:
            return False, "Global kill switch is active"
        
        # Check playbook kill switch
        if action.incident_id in self.playbook_kill_switches:
            if self.playbook_kill_switches.get(action.incident_id):
                return False, f"Playbook kill switch active for incident"
        
        # Check scope kill switch
        if self.scope_kill_switches.get(action.target):
            return False, f"Kill switch active for target: {action.target}"
        
        # Check prohibited actions
        if action.action_type in self.prohibited_actions:
            return False, f"Action type '{action.action_type}' is prohibited"
        
        # Check approval requirements
        if action.action_type in self.approval_required and not action.approved_by:
            return False, f"Action '{action.action_type}' requires approval"
        
        # Check rate limits
        if action.action_type in self.rate_limits:
            can_proceed, reason = self._check_rate_limit(action.action_type)
            if not can_proceed:
                return False, reason
        
        # Check blast radius
        can_proceed, reason = self._check_blast_radius(action)
        if not can_proceed:
            return False, reason
        
        return True, "Safety checks passed"
    
    def _check_rate_limit(self, action_type: str) -> Tuple[bool, str]:
        """Check if action is within rate limits"""
        limit = self.rate_limits.get(action_type, 100)
        
        # Clean old entries
        cutoff = datetime.now() - timedelta(hours=1)
        self.action_counts[action_type] = [
            t for t in self.action_counts[action_type]
            if t > cutoff
        ]
        
        if len(self.action_counts[action_type]) >= limit:
            return False, f"Rate limit exceeded for {action_type}: {limit}/hour"
        
        return True, "Within rate limits"
    
    def _check_blast_radius(self, action: ResponseAction) -> Tuple[bool, str]:
        """Check if action exceeds blast radius limits"""
        if action.action_type == 'isolate_host':
            if len(self.affected_entities['isolated_hosts']) >= self.blast_radius_limits['max_hosts_isolated']:
                return False, f"Blast radius limit: max {self.blast_radius_limits['max_hosts_isolated']} hosts isolated"
        
        if action.action_type == 'disable_account':
            if len(self.affected_entities['disabled_accounts']) >= self.blast_radius_limits['max_accounts_disabled']:
                return False, f"Blast radius limit: max {self.blast_radius_limits['max_accounts_disabled']} accounts disabled"
        
        return True, "Within blast radius"
    
    def record_action(self, action: ResponseAction):
        """Record executed action for tracking"""
        self.action_counts[action.action_type].append(datetime.now())
        
        if action.action_type == 'isolate_host':
            self.affected_entities['isolated_hosts'].add(action.target)
        elif action.action_type == 'disable_account':
            self.affected_entities['disabled_accounts'].add(action.target)
        elif action.action_type == 'block_ip':
            self.affected_entities['blocked_ips'].add(action.target)
    
    def activate_kill_switch(self, scope: str = 'global'):
        """Activate kill switch"""
        if scope == 'global':
            self.global_kill_switch = True
        else:
            self.scope_kill_switches[scope] = True
    
    def deactivate_kill_switch(self, scope: str = 'global'):
        """Deactivate kill switch"""
        if scope == 'global':
            self.global_kill_switch = False
        else:
            self.scope_kill_switches[scope] = False


class EvidenceCollector:
    """
    Collects and preserves evidence for forensic investigation.
    """
    
    def __init__(self):
        """Initialize evidence collector"""
        self.evidence_store: Dict[str, Dict[str, Any]] = {}
    
    def collect_evidence(self,
                        incident_id: str,
                        evidence_type: str,
                        data: Any,
                        source: str) -> str:
        """
        Collect and hash evidence.
        
        Returns:
            Evidence ID
        """
        evidence_id = f"evidence_{incident_id}_{datetime.now().timestamp()}"
        
        # Hash the evidence for integrity
        if isinstance(data, str):
            content_hash = hashlib.sha256(data.encode()).hexdigest()
        else:
            content_hash = hashlib.sha256(json.dumps(data, default=str).encode()).hexdigest()
        
        evidence_record = {
            'evidence_id': evidence_id,
            'incident_id': incident_id,
            'evidence_type': evidence_type,
            'collected_at': datetime.now(),
            'source': source,
            'content_hash': content_hash,
            'data': data
        }
        
        if incident_id not in self.evidence_store:
            self.evidence_store[incident_id] = {}
        
        self.evidence_store[incident_id][evidence_id] = evidence_record
        
        return evidence_id
    
    def get_evidence(self, incident_id: str) -> List[Dict[str, Any]]:
        """Get all evidence for an incident"""
        return list(self.evidence_store.get(incident_id, {}).values())
    
    def verify_evidence_integrity(self, incident_id: str, evidence_id: str) -> bool:
        """Verify evidence hasn't been tampered with"""
        evidence = self.evidence_store.get(incident_id, {}).get(evidence_id)
        
        if not evidence:
            return False
        
        data = evidence['data']
        if isinstance(data, str):
            current_hash = hashlib.sha256(data.encode()).hexdigest()
        else:
            current_hash = hashlib.sha256(json.dumps(data, default=str).encode()).hexdigest()
        
        return current_hash == evidence['content_hash']


class ActionExecutor:
    """
    Executes response actions with safety checks and logging.
    """
    
    def __init__(self, safety_controller: SafetyController, evidence_collector: EvidenceCollector):
        """Initialize action executor"""
        self.safety_controller = safety_controller
        self.evidence_collector = evidence_collector
        self.execution_log: List[Dict[str, Any]] = []
    
    def execute_action(self, action: ResponseAction) -> ResponseAction:
        """
        Execute a response action with safety checks.
        
        Args:
            action: Action to execute
            
        Returns:
            Updated action with execution result
        """
        # Safety check
        can_execute, reason = self.safety_controller.can_execute(action)
        
        if not can_execute:
            action.status = ActionStatus.BLOCKED
            action.error_message = reason
            self._log_execution(action, success=False, reason=reason)
            return action
        
        # Execute action
        action.status = ActionStatus.EXECUTING
        action.executed_at = datetime.now()
        
        try:
            result = self._execute(action)
            action.result = result
            action.status = ActionStatus.COMPLETED
            
            # Record for safety tracking
            self.safety_controller.record_action(action)
            
            self._log_execution(action, success=True)
            
        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
            self._log_execution(action, success=False, reason=str(e))
        
        return action
    
    def _execute(self, action: ResponseAction) -> Dict[str, Any]:
        """
        Execute the specific action type.
        
        In production, this would call actual infrastructure APIs.
        """
        # Simulate execution based on action type
        result = {
            'action_type': action.action_type,
            'target': action.target,
            'executed_at': datetime.now().isoformat(),
            'simulated': True
        }
        
        if action.action_type == 'isolate_host':
            result['network_isolation'] = 'applied'
            result['previous_state'] = 'connected'
        
        elif action.action_type == 'disable_account':
            result['account_status'] = 'disabled'
            result['sessions_terminated'] = 3
        
        elif action.action_type == 'block_ip':
            result['firewall_rule'] = 'created'
            result['blocked_ip'] = action.target
        
        elif action.action_type == 'quarantine_email':
            result['email_status'] = 'quarantined'
            result['removed_from_mailboxes'] = 15
        
        elif action.action_type == 'force_password_reset':
            result['password_reset'] = 'required'
            result['sessions_invalidated'] = True
        
        elif action.action_type == 'collect_forensics':
            result['memory_dump'] = 'collected'
            result['disk_snapshot'] = 'created'
            result['evidence_preserved'] = True
        
        elif action.action_type == 'terminate_process':
            result['process_terminated'] = True
            result['pid'] = action.parameters.get('pid', 'unknown')
        
        return result
    
    def rollback_action(self, action: ResponseAction) -> ResponseAction:
        """Rollback a previously executed action"""
        if not action.rollback_action:
            action.error_message = "No rollback action defined"
            return action
        
        rollback = action.rollback_action
        rollback.status = ActionStatus.EXECUTING
        
        try:
            result = self._execute(rollback)
            rollback.result = result
            rollback.status = ActionStatus.COMPLETED
            action.status = ActionStatus.ROLLED_BACK
            
        except Exception as e:
            rollback.status = ActionStatus.FAILED
            rollback.error_message = str(e)
        
        return action
    
    def _log_execution(self, action: ResponseAction, success: bool, reason: str = None):
        """Log action execution"""
        self.execution_log.append({
            'timestamp': datetime.now(),
            'action_id': action.action_id,
            'action_type': action.action_type,
            'target': action.target,
            'success': success,
            'reason': reason
        })


class PlaybookEngine:
    """
    Executes security response playbooks.
    """
    
    def __init__(self, action_executor: ActionExecutor, bedrock_client):
        """Initialize playbook engine"""
        self.action_executor = action_executor
        self.bedrock_client = bedrock_client
        self.playbooks: Dict[str, ResponsePlaybook] = {}
        self.pending_approvals: List[ResponseAction] = []
        
        self._register_standard_playbooks()
    
    def _register_standard_playbooks(self):
        """Register standard security playbooks"""
        
        # Phishing Response Playbook
        self.playbooks['phishing'] = ResponsePlaybook(
            playbook_id='phishing',
            name='Phishing Response',
            description='Automated response to phishing incidents',
            trigger_types=['phishing', 'suspicious_email'],
            severity_threshold=IncidentSeverity.MEDIUM,
            steps=[
                PlaybookStep('ph_1', 'Quarantine Email', 'Quarantine the phishing email',
                            'quarantine_email', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('ph_2', 'Extract Indicators', 'Extract URLs and attachment hashes',
                            'extract_indicators', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('ph_3', 'Check Recipients', 'Identify all recipients of the email',
                            'find_recipients', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('ph_4', 'Block Sender', 'Block the sender domain/address',
                            'block_sender', ActionCategory.CONTAINMENT, AutomationLevel.HUMAN_APPROVED),
                PlaybookStep('ph_5', 'Remove from Mailboxes', 'Remove email from all mailboxes',
                            'remove_emails', ActionCategory.ERADICATION, AutomationLevel.BOUNDED),
                PlaybookStep('ph_6', 'Notify Users', 'Alert affected users',
                            'notify_users', ActionCategory.NOTIFICATION, AutomationLevel.BOUNDED),
            ]
        )
        
        # Malware Response Playbook
        self.playbooks['malware'] = ResponsePlaybook(
            playbook_id='malware',
            name='Malware Response',
            description='Automated response to malware detection',
            trigger_types=['malware', 'ransomware', 'trojan'],
            severity_threshold=IncidentSeverity.HIGH,
            steps=[
                PlaybookStep('mw_1', 'Isolate Host', 'Immediately isolate infected host',
                            'isolate_host', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('mw_2', 'Terminate Process', 'Kill malicious process',
                            'terminate_process', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('mw_3', 'Collect Forensics', 'Capture memory and disk evidence',
                            'collect_forensics', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('mw_4', 'Check Lateral Movement', 'Scan for spread to other hosts',
                            'check_lateral_movement', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('mw_5', 'Block C2', 'Block command and control communications',
                            'block_ip', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('mw_6', 'Notify IT', 'Alert IT team for remediation',
                            'notify_it', ActionCategory.NOTIFICATION, AutomationLevel.BOUNDED),
            ]
        )
        
        # Compromised Credential Playbook
        self.playbooks['credential_compromise'] = ResponsePlaybook(
            playbook_id='credential_compromise',
            name='Compromised Credential Response',
            description='Response to credential abuse or theft',
            trigger_types=['credential_compromise', 'impossible_travel', 'account_takeover'],
            severity_threshold=IncidentSeverity.HIGH,
            steps=[
                PlaybookStep('cc_1', 'Terminate Sessions', 'Invalidate all active sessions',
                            'terminate_sessions', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('cc_2', 'Force Password Reset', 'Require immediate password change',
                            'force_password_reset', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('cc_3', 'Disable Account', 'Temporarily disable the account',
                            'disable_account', ActionCategory.CONTAINMENT, AutomationLevel.HUMAN_APPROVED),
                PlaybookStep('cc_4', 'Review Activity', 'Collect recent account activity',
                            'collect_activity_logs', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('cc_5', 'Check for Changes', 'Review unauthorized changes made',
                            'audit_changes', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('cc_6', 'Require MFA Re-enrollment', 'Force MFA setup again',
                            'require_mfa', ActionCategory.ERADICATION, AutomationLevel.BOUNDED),
                PlaybookStep('cc_7', 'Notify User', 'Contact user via alternate channel',
                            'notify_user', ActionCategory.NOTIFICATION, AutomationLevel.BOUNDED),
            ]
        )
        
        # Data Exfiltration Playbook
        self.playbooks['exfiltration'] = ResponsePlaybook(
            playbook_id='exfiltration',
            name='Data Exfiltration Response',
            description='Response to data theft detection',
            trigger_types=['data_exfiltration', 'dlp_violation', 'unusual_download'],
            severity_threshold=IncidentSeverity.CRITICAL,
            steps=[
                PlaybookStep('ex_1', 'Block Transfer', 'Stop ongoing data transfer',
                            'block_transfer', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('ex_2', 'Isolate Source', 'Isolate the source system',
                            'isolate_host', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('ex_3', 'Disable External Access', 'Revoke external access',
                            'disable_external_access', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('ex_4', 'Identify Data', 'Determine what data was accessed',
                            'identify_data', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('ex_5', 'Preserve Evidence', 'Collect forensic evidence',
                            'collect_forensics', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('ex_6', 'Escalate to Legal', 'Notify legal team',
                            'escalate_legal', ActionCategory.NOTIFICATION, AutomationLevel.BOUNDED),
            ]
        )
        
        # Ransomware Response Playbook
        self.playbooks['ransomware'] = ResponsePlaybook(
            playbook_id='ransomware',
            name='Ransomware Response',
            description='Emergency response to ransomware attack',
            trigger_types=['ransomware', 'encryption_detected'],
            severity_threshold=IncidentSeverity.CRITICAL,
            steps=[
                PlaybookStep('rw_1', 'Immediate Isolation', 'Isolate infected host NOW',
                            'isolate_host', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('rw_2', 'Segment Network', 'Isolate network segment',
                            'isolate_segment', ActionCategory.CONTAINMENT, AutomationLevel.HUMAN_APPROVED),
                PlaybookStep('rw_3', 'Disable Accounts', 'Disable impacted user accounts',
                            'disable_account', ActionCategory.CONTAINMENT, AutomationLevel.BOUNDED),
                PlaybookStep('rw_4', 'Preserve Evidence', 'Capture forensic evidence',
                            'collect_forensics', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('rw_5', 'Identify Patient Zero', 'Find initial infection point',
                            'find_patient_zero', ActionCategory.EVIDENCE, AutomationLevel.BOUNDED),
                PlaybookStep('rw_6', 'Verify Backups', 'Check backup integrity',
                            'verify_backups', ActionCategory.RECOVERY, AutomationLevel.BOUNDED),
                PlaybookStep('rw_7', 'Activate IR Team', 'Full incident response activation',
                            'activate_ir_team', ActionCategory.NOTIFICATION, AutomationLevel.BOUNDED),
            ]
        )
    
    def get_playbook(self, incident_type: str) -> Optional[ResponsePlaybook]:
        """Get playbook for incident type"""
        for playbook in self.playbooks.values():
            if incident_type in playbook.trigger_types and playbook.enabled:
                return playbook
        return None
    
    def execute_playbook(self,
                        incident: SecurityIncident,
                        playbook: ResponsePlaybook) -> List[ResponseAction]:
        """
        Execute a playbook for an incident.
        
        Args:
            incident: The security incident
            playbook: Playbook to execute
            
        Returns:
            List of executed actions
        """
        executed_actions = []
        
        incident.playbook_id = playbook.playbook_id
        incident.status = IncidentStatus.CONTAINING
        
        for step in playbook.steps:
            # Check if step condition is met
            if step.condition and not step.condition(incident):
                continue
            
            # Create action from step
            action = ResponseAction(
                action_id=f"action_{incident.incident_id}_{step.step_id}",
                incident_id=incident.incident_id,
                timestamp=datetime.now(),
                action_type=step.action_type,
                category=step.category,
                target=incident.affected_entities[0] if incident.affected_entities else 'unknown',
                parameters=step.parameters,
                automation_level=step.automation_level,
                status=ActionStatus.PENDING,
                requires_approval=(step.automation_level == AutomationLevel.HUMAN_APPROVED)
            )
            
            # Handle based on automation level
            if step.automation_level == AutomationLevel.HUMAN_APPROVED:
                action.status = ActionStatus.PENDING
                self.pending_approvals.append(action)
                executed_actions.append(action)
                
                # Add to timeline
                incident.timeline.append({
                    'timestamp': datetime.now(),
                    'event': f"Action pending approval: {step.name}",
                    'action_id': action.action_id
                })
            
            elif step.automation_level in [AutomationLevel.BOUNDED, AutomationLevel.ADAPTIVE]:
                # Execute automatically
                action = self.action_executor.execute_action(action)
                executed_actions.append(action)
                
                # Add to timeline
                incident.timeline.append({
                    'timestamp': datetime.now(),
                    'event': f"Action executed: {step.name}",
                    'action_id': action.action_id,
                    'status': action.status.value
                })
                
                # Check if we should continue
                if action.status == ActionStatus.FAILED and not step.continue_on_failure:
                    incident.timeline.append({
                        'timestamp': datetime.now(),
                        'event': f"Playbook halted due to failed action: {step.name}"
                    })
                    break
        
        return executed_actions
    
    def approve_action(self, action_id: str, approver: str) -> Optional[ResponseAction]:
        """Approve a pending action"""
        for action in self.pending_approvals:
            if action.action_id == action_id:
                action.approved_by = approver
                action.status = ActionStatus.APPROVED
                
                # Execute the action
                action = self.action_executor.execute_action(action)
                
                self.pending_approvals.remove(action)
                return action
        
        return None
    
    def reject_action(self, action_id: str, reason: str) -> Optional[ResponseAction]:
        """Reject a pending action"""
        for action in self.pending_approvals:
            if action.action_id == action_id:
                action.status = ActionStatus.BLOCKED
                action.error_message = f"Rejected: {reason}"
                self.pending_approvals.remove(action)
                return action
        
        return None


class IncidentResponsePipeline:
    """
    Production pipeline orchestrating automated incident response.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize incident response pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.safety_controller = SafetyController()
        self.evidence_collector = EvidenceCollector()
        self.action_executor = ActionExecutor(self.safety_controller, self.evidence_collector)
        self.playbook_engine = PlaybookEngine(self.action_executor, self.bedrock_client)
        
        self.incidents: Dict[str, SecurityIncident] = {}
        self.escalation_paths = self._configure_escalation()
    
    def _configure_escalation(self) -> Dict[str, List[EscalationPath]]:
        """Configure escalation paths by severity"""
        return {
            'critical': [
                EscalationPath(1, 'SOC Analyst', 'slack', 5),
                EscalationPath(2, 'SOC Manager', 'page', 10),
                EscalationPath(3, 'CISO', 'page', 15),
            ],
            'high': [
                EscalationPath(1, 'SOC Analyst', 'slack', 15),
                EscalationPath(2, 'SOC Manager', 'slack', 30),
            ],
            'medium': [
                EscalationPath(1, 'SOC Analyst', 'email', 60),
            ],
            'low': [
                EscalationPath(1, 'SOC Analyst', 'email', 120),
            ]
        }
    
    def handle_incident(self, incident: SecurityIncident) -> Dict[str, Any]:
        """
        Handle a security incident with automated response.
        
        Args:
            incident: Security incident to handle
            
        Returns:
            Response summary
        """
        # Store incident
        self.incidents[incident.incident_id] = incident
        
        # Add to timeline
        incident.timeline.append({
            'timestamp': datetime.now(),
            'event': 'Incident detected and response initiated'
        })
        
        # Get appropriate playbook
        playbook = self.playbook_engine.get_playbook(incident.incident_type)
        
        if not playbook:
            incident.status = IncidentStatus.ESCALATED
            incident.timeline.append({
                'timestamp': datetime.now(),
                'event': 'No playbook found - escalating to analyst'
            })
            return self._create_response_summary(incident, [], 'No playbook available')
        
        # Check severity threshold
        severity_order = [IncidentSeverity.LOW, IncidentSeverity.MEDIUM, 
                        IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
        
        if severity_order.index(incident.severity) < severity_order.index(playbook.severity_threshold):
            incident.timeline.append({
                'timestamp': datetime.now(),
                'event': f'Severity below playbook threshold - manual handling required'
            })
            return self._create_response_summary(incident, [], 'Below severity threshold')
        
        # Execute playbook
        incident.timeline.append({
            'timestamp': datetime.now(),
            'event': f'Executing playbook: {playbook.name}'
        })
        
        executed_actions = self.playbook_engine.execute_playbook(incident, playbook)
        
        # Collect evidence
        for entity in incident.affected_entities:
            self.evidence_collector.collect_evidence(
                incident.incident_id,
                'entity_activity',
                {'entity': entity, 'incident_type': incident.incident_type},
                'automated_collection'
            )
        
        # Update status based on actions
        successful_actions = [a for a in executed_actions if a.status == ActionStatus.COMPLETED]
        pending_actions = [a for a in executed_actions if a.status == ActionStatus.PENDING]
        
        if pending_actions:
            incident.status = IncidentStatus.TRIAGING
        elif successful_actions:
            incident.status = IncidentStatus.ERADICATING
        
        incident.timeline.append({
            'timestamp': datetime.now(),
            'event': f'Playbook execution completed: {len(successful_actions)} actions completed, {len(pending_actions)} pending approval'
        })
        
        # Generate AI analysis
        ai_analysis = self._generate_incident_analysis(incident, executed_actions)
        
        return self._create_response_summary(incident, executed_actions, ai_analysis)
    
    def _create_response_summary(self,
                                incident: SecurityIncident,
                                actions: List[ResponseAction],
                                analysis: str) -> Dict[str, Any]:
        """Create response summary"""
        return {
            'incident_id': incident.incident_id,
            'incident_type': incident.incident_type,
            'severity': incident.severity.value,
            'status': incident.status.value,
            'actions_executed': len([a for a in actions if a.status == ActionStatus.COMPLETED]),
            'actions_pending': len([a for a in actions if a.status == ActionStatus.PENDING]),
            'actions_failed': len([a for a in actions if a.status == ActionStatus.FAILED]),
            'affected_entities': incident.affected_entities,
            'evidence_collected': len(self.evidence_collector.get_evidence(incident.incident_id)),
            'analysis': analysis,
            'timeline_events': len(incident.timeline)
        }
    
    def _generate_incident_analysis(self,
                                   incident: SecurityIncident,
                                   actions: List[ResponseAction]) -> str:
        """Generate AI-powered incident analysis"""
        actions_summary = "\n".join([
            f"- {a.action_type}: {a.status.value}"
            for a in actions[:10]
        ])
        
        timeline_summary = "\n".join([
            f"- [{e['timestamp']}] {e['event']}"
            for e in incident.timeline[-5:]
        ])
        
        prompt = f"""Analyze this security incident and response:

Incident Type: {incident.incident_type}
Severity: {incident.severity.value}
Affected Entities: {', '.join(incident.affected_entities)}
Status: {incident.status.value}

Actions Taken:
{actions_summary}

Recent Timeline:
{timeline_summary}

Provide:
1. Brief assessment of the response effectiveness (2 sentences)
2. Any gaps in the response
3. Recommended next steps for analysts

Be concise and actionable."""

        try:
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 400,
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
    
    def get_pending_approvals(self) -> List[ResponseAction]:
        """Get all actions pending approval"""
        return self.playbook_engine.pending_approvals
    
    def approve_action(self, action_id: str, approver: str) -> Optional[ResponseAction]:
        """Approve a pending action"""
        action = self.playbook_engine.approve_action(action_id, approver)
        
        if action:
            # Update incident timeline
            incident = self.incidents.get(action.incident_id)
            if incident:
                incident.timeline.append({
                    'timestamp': datetime.now(),
                    'event': f'Action approved by {approver}: {action.action_type}',
                    'action_id': action.action_id
                })
        
        return action
    
    def activate_kill_switch(self, scope: str = 'global'):
        """Activate kill switch to stop automation"""
        self.safety_controller.activate_kill_switch(scope)
    
    def get_incident_report(self, incident_id: str) -> str:
        """Generate detailed incident report"""
        incident = self.incidents.get(incident_id)
        
        if not incident:
            return "Incident not found"
        
        evidence = self.evidence_collector.get_evidence(incident_id)
        
        timeline_text = "\n".join([
            f"  [{e['timestamp']}] {e['event']}"
            for e in incident.timeline
        ])
        
        prompt = f"""Generate a formal security incident report:

Incident ID: {incident.incident_id}
Type: {incident.incident_type}
Severity: {incident.severity.value}
Status: {incident.status.value}
Affected Entities: {', '.join(incident.affected_entities)}
Indicators: {', '.join(incident.indicators)}

Timeline:
{timeline_text}

Evidence Items: {len(evidence)}

Generate a structured incident report with:
1. Executive Summary
2. Incident Details
3. Response Actions Taken
4. Impact Assessment
5. Recommendations

Format as a professional security incident report."""

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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get incident response statistics"""
        if not self.incidents:
            return {'no_incidents': True}
        
        incidents_list = list(self.incidents.values())
        
        by_type = Counter(i.incident_type for i in incidents_list)
        by_severity = Counter(i.severity.value for i in incidents_list)
        by_status = Counter(i.status.value for i in incidents_list)
        
        return {
            'total_incidents': len(incidents_list),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'by_status': dict(by_status),
            'pending_approvals': len(self.playbook_engine.pending_approvals),
            'actions_executed': len(self.action_executor.execution_log),
            'evidence_items': sum(
                len(self.evidence_collector.get_evidence(i.incident_id))
                for i in incidents_list
            )
        }


# Example usage and realistic incident scenarios
def create_sample_incidents() -> List[SecurityIncident]:
    """Create sample security incidents for demonstration"""
    incidents = []
    
    # Phishing incident
    incidents.append(SecurityIncident(
        incident_id='INC-2024-001',
        timestamp=datetime.now(),
        incident_type='phishing',
        severity=IncidentSeverity.MEDIUM,
        status=IncidentStatus.DETECTED,
        source_alert_id='ALERT-PH-001',
        affected_entities=['jsmith@company.com', 'ajones@company.com'],
        indicators=['malicious-site.com', 'phishing@attacker.net'],
        description='Phishing email detected targeting multiple employees with credential harvesting link'
    ))
    
    # Malware incident
    incidents.append(SecurityIncident(
        incident_id='INC-2024-002',
        timestamp=datetime.now(),
        incident_type='malware',
        severity=IncidentSeverity.HIGH,
        status=IncidentStatus.DETECTED,
        source_alert_id='ALERT-MW-001',
        affected_entities=['WORKSTATION-PC42'],
        indicators=['185.143.223.47', 'trojan.exe', 'c2-beacon.dll'],
        description='Malware detected on workstation with active C2 communication'
    ))
    
    # Credential compromise incident
    incidents.append(SecurityIncident(
        incident_id='INC-2024-003',
        timestamp=datetime.now(),
        incident_type='credential_compromise',
        severity=IncidentSeverity.HIGH,
        status=IncidentStatus.DETECTED,
        source_alert_id='ALERT-UEBA-001',
        affected_entities=['bwilson'],
        indicators=['185.143.223.100', 'Moscow, Russia'],
        description='Impossible travel detected - user login from Russia while session active in USA'
    ))
    
    # Ransomware incident
    incidents.append(SecurityIncident(
        incident_id='INC-2024-004',
        timestamp=datetime.now(),
        incident_type='ransomware',
        severity=IncidentSeverity.CRITICAL,
        status=IncidentStatus.DETECTED,
        source_alert_id='ALERT-RW-001',
        affected_entities=['FILE-SERVER-01', 'admin_user'],
        indicators=['encrypt.exe', '.locked extension', 'ransom_note.txt'],
        description='Ransomware activity detected - file encryption in progress on file server'
    ))
    
    return incidents


def main():
    """
    Demonstrate automated incident response pipeline.
    """
    print("=" * 80)
    print("Chapter 19: Automated Security Incident Response")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing incident response pipeline...")
    pipeline = IncidentResponsePipeline(aws_region='us-east-1')
    print("✓ Pipeline initialized")
    print(f"✓ Loaded {len(pipeline.playbook_engine.playbooks)} response playbooks")
    print(f"✓ Safety controls configured")
    print()
    
    # Create sample incidents
    incidents = create_sample_incidents()
    
    # Process each incident
    for incident in incidents:
        print("=" * 80)
        print(f"INCIDENT: {incident.incident_id}")
        print("=" * 80)
        print()
        
        print(f"Type: {incident.incident_type}")
        print(f"Severity: {incident.severity.value}")
        print(f"Affected: {', '.join(incident.affected_entities)}")
        print(f"Description: {incident.description}")
        print()
        
        # Handle incident
        response = pipeline.handle_incident(incident)
        
        print("RESPONSE SUMMARY:")
        print(f"  Status: {response['status']}")
        print(f"  Actions Executed: {response['actions_executed']}")
        print(f"  Actions Pending: {response['actions_pending']}")
        print(f"  Actions Failed: {response['actions_failed']}")
        print(f"  Evidence Collected: {response['evidence_collected']}")
        print()
        
        print("AI ANALYSIS:")
        print(response['analysis'])
        print()
        
        # Show timeline
        print("TIMELINE:")
        for event in incident.timeline[-5:]:
            print(f"  [{event['timestamp'].strftime('%H:%M:%S')}] {event['event']}")
        print()
    
    # Show pending approvals
    print("=" * 80)
    print("PENDING APPROVALS")
    print("=" * 80)
    print()
    
    pending = pipeline.get_pending_approvals()
    if pending:
        for action in pending:
            print(f"  [{action.action_id}] {action.action_type}")
            print(f"    Target: {action.target}")
            print(f"    Incident: {action.incident_id}")
            print()
        
        # Simulate approval
        if pending:
            print("Simulating approval for first pending action...")
            approved = pipeline.approve_action(pending[0].action_id, 'analyst@company.com')
            if approved:
                print(f"  ✓ Action approved and executed: {approved.status.value}")
    else:
        print("  No pending approvals")
    print()
    
    # Generate incident report for ransomware incident
    print("=" * 80)
    print("INCIDENT REPORT: INC-2024-004 (Ransomware)")
    print("=" * 80)
    print()
    
    report = pipeline.get_incident_report('INC-2024-004')
    print(report)
    print()
    
    # Statistics
    print("=" * 80)
    print("RESPONSE STATISTICS")
    print("=" * 80)
    print()
    
    stats = pipeline.get_statistics()
    print(f"Total Incidents: {stats['total_incidents']}")
    print(f"Pending Approvals: {stats['pending_approvals']}")
    print(f"Actions Executed: {stats['actions_executed']}")
    print(f"Evidence Items: {stats['evidence_items']}")
    
    print(f"\nBy Type:")
    for itype, count in stats['by_type'].items():
        print(f"  {itype}: {count}")
    
    print(f"\nBy Severity:")
    for severity, count in stats['by_severity'].items():
        print(f"  {severity}: {count}")
    
    print(f"\nBy Status:")
    for status, count in stats['by_status'].items():
        print(f"  {status}: {count}")
    
    print("\n" + "=" * 80)
    print("Automated Incident Response Demonstration Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
