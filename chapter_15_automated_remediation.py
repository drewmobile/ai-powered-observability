"""
Chapter 15: Automated Remediation
AI-Powered Observability: From Noise to Insight

This module demonstrates production-ready automated remediation including:
- Rule-based remediation actions
- Runbook automation and orchestration
- Self-healing infrastructure patterns
- Progressive remediation (escalating actions)
- AI-enhanced pattern recognition for action selection
- Safety mechanisms (circuit breakers, rate limiting, rollback)
- Remediation verification and monitoring
- AI-powered remediation decisions with AWS Bedrock

Author: AI-Powered Observability
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
warnings.filterwarnings('ignore')


class RemediationStatus(Enum):
    """Status of a remediation execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ESCALATED = "escalated"


class RemediationSafety(Enum):
    """Safety level of remediation action"""
    SAFE = "safe"  # Low risk, reversible
    MODERATE = "moderate"  # Some risk, requires verification
    RISKY = "risky"  # High risk, requires approval


@dataclass
class Problem:
    """Represents a detected problem requiring remediation"""
    problem_id: str
    timestamp: datetime
    service: str
    problem_type: str  # process_crash, memory_leak, disk_full, etc.
    severity: str  # critical, high, medium, low
    symptoms: Dict[str, Any]
    detected_by: str  # monitoring system that detected it


@dataclass
class RemediationAction:
    """Represents a remediation action"""
    action_id: str
    action_type: str  # restart, scale, cleanup, rollback, etc.
    target: str  # Service or component to act on
    parameters: Dict[str, Any]
    safety_level: RemediationSafety
    estimated_duration_seconds: int
    rollback_action: Optional['RemediationAction'] = None
    requires_approval: bool = False


@dataclass
class RemediationExecution:
    """Represents execution of a remediation"""
    execution_id: str
    problem: Problem
    action: RemediationAction
    status: RemediationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    verification_result: Optional[Dict[str, Any]] = None
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunbookStep:
    """Represents a step in an automated runbook"""
    step_id: str
    description: str
    action: Callable
    verification: Optional[Callable] = None
    rollback: Optional[Callable] = None
    timeout_seconds: int = 60
    continue_on_failure: bool = False


class CircuitBreaker:
    """
    Circuit breaker to prevent runaway automation.
    """
    
    def __init__(self, failure_threshold: int = 3, timeout_seconds: int = 300):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failures = defaultdict(int)
        self.opened_at: Dict[str, datetime] = {}
        self.is_open: Dict[str, bool] = defaultdict(bool)
    
    def record_success(self, action_type: str):
        """Record successful execution"""
        self.failures[action_type] = 0
        self.is_open[action_type] = False
    
    def record_failure(self, action_type: str):
        """Record failed execution"""
        self.failures[action_type] += 1
        
        if self.failures[action_type] >= self.failure_threshold:
            self.is_open[action_type] = True
            self.opened_at[action_type] = datetime.now()
    
    def can_execute(self, action_type: str) -> Tuple[bool, str]:
        """
        Check if action can be executed.
        
        Returns:
            Tuple of (can_execute, reason)
        """
        if not self.is_open.get(action_type, False):
            return True, "Circuit closed"
        
        # Check if timeout has passed
        opened = self.opened_at.get(action_type)
        if opened and (datetime.now() - opened).total_seconds() > self.timeout_seconds:
            self.is_open[action_type] = False
            self.failures[action_type] = 0
            return True, "Circuit timeout expired, attempting close"
        
        return False, f"Circuit open: {self.failures[action_type]} consecutive failures"


class RateLimiter:
    """
    Rate limiter to prevent excessive automation.
    """
    
    def __init__(self, max_actions_per_hour: int = 10):
        """
        Initialize rate limiter.
        
        Args:
            max_actions_per_hour: Maximum actions allowed per hour
        """
        self.max_actions_per_hour = max_actions_per_hour
        self.action_times: Dict[str, List[datetime]] = defaultdict(list)
    
    def can_execute(self, action_type: str) -> Tuple[bool, str]:
        """
        Check if action can be executed within rate limits.
        
        Returns:
            Tuple of (can_execute, reason)
        """
        # Clean old entries (older than 1 hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.action_times[action_type] = [
            t for t in self.action_times[action_type]
            if t > cutoff
        ]
        
        if len(self.action_times[action_type]) >= self.max_actions_per_hour:
            return False, f"Rate limit exceeded: {len(self.action_times[action_type])} actions in last hour"
        
        return True, "Within rate limits"
    
    def record_execution(self, action_type: str):
        """Record action execution"""
        self.action_times[action_type].append(datetime.now())


class RemediationActionLibrary:
    """
    Library of available remediation actions.
    """
    
    def __init__(self):
        """Initialize action library"""
        self.actions: Dict[str, RemediationAction] = {}
        self._register_standard_actions()
    
    def _register_standard_actions(self):
        """Register standard remediation actions"""
        
        # Restart service
        self.register_action(RemediationAction(
            action_id='restart_service',
            action_type='restart',
            target='service',
            parameters={},
            safety_level=RemediationSafety.SAFE,
            estimated_duration_seconds=30,
            requires_approval=False
        ))
        
        # Scale up
        self.register_action(RemediationAction(
            action_id='scale_up',
            action_type='scale',
            target='service',
            parameters={'direction': 'up', 'increment': 1},
            safety_level=RemediationSafety.SAFE,
            estimated_duration_seconds=60,
            requires_approval=False
        ))
        
        # Clear cache
        self.register_action(RemediationAction(
            action_id='clear_cache',
            action_type='cleanup',
            target='cache',
            parameters={},
            safety_level=RemediationSafety.SAFE,
            estimated_duration_seconds=10,
            requires_approval=False
        ))
        
        # Cleanup disk
        self.register_action(RemediationAction(
            action_id='cleanup_disk',
            action_type='cleanup',
            target='disk',
            parameters={'rotate_logs': True, 'delete_temp': True},
            safety_level=RemediationSafety.SAFE,
            estimated_duration_seconds=30,
            requires_approval=False
        ))
        
        # Rollback deployment
        self.register_action(RemediationAction(
            action_id='rollback_deployment',
            action_type='rollback',
            target='service',
            parameters={'target_version': 'previous'},
            safety_level=RemediationSafety.MODERATE,
            estimated_duration_seconds=120,
            requires_approval=True
        ))
        
        # Increase connection pool
        self.register_action(RemediationAction(
            action_id='increase_connection_pool',
            action_type='config_change',
            target='database',
            parameters={'pool_size_increment': 10},
            safety_level=RemediationSafety.MODERATE,
            estimated_duration_seconds=20,
            requires_approval=False
        ))
    
    def register_action(self, action: RemediationAction):
        """Register a new remediation action"""
        self.actions[action.action_id] = action
    
    def get_action(self, action_id: str) -> Optional[RemediationAction]:
        """Get action by ID"""
        return self.actions.get(action_id)
    
    def get_actions_for_problem(self, problem_type: str) -> List[RemediationAction]:
        """Get recommended actions for a problem type"""
        # Simplified mapping
        mappings = {
            'process_crash': ['restart_service'],
            'memory_leak': ['restart_service', 'scale_up'],
            'disk_full': ['cleanup_disk'],
            'high_latency': ['restart_service', 'scale_up', 'clear_cache'],
            'connection_exhaustion': ['increase_connection_pool', 'restart_service'],
            'deployment_issue': ['rollback_deployment']
        }
        
        action_ids = mappings.get(problem_type, [])
        return [self.get_action(aid) for aid in action_ids if self.get_action(aid)]


class PatternBasedRemediationSelector:
    """
    AI-enhanced selector that learns which remediations work for which patterns.
    """
    
    def __init__(self, bedrock_client):
        """Initialize pattern-based selector"""
        self.bedrock_client = bedrock_client
        self.pattern_history: List[Dict[str, Any]] = []
    
    def record_remediation_result(self,
                                  problem: Problem,
                                  action: RemediationAction,
                                  success: bool):
        """Record the outcome of a remediation"""
        self.pattern_history.append({
            'problem_type': problem.problem_type,
            'symptoms': problem.symptoms,
            'action_id': action.action_id,
            'success': success,
            'timestamp': datetime.now()
        })
        
        # Keep last 1000 records
        if len(self.pattern_history) > 1000:
            self.pattern_history = self.pattern_history[-1000:]
    
    def select_remediation(self,
                          problem: Problem,
                          available_actions: List[RemediationAction]) -> Optional[RemediationAction]:
        """
        Select best remediation action based on learned patterns.
        
        Args:
            problem: Problem to remediate
            available_actions: Available remediation actions
            
        Returns:
            Recommended action
        """
        # Calculate success rates for each action with this problem type
        action_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        
        for record in self.pattern_history:
            if record['problem_type'] == problem.problem_type:
                action_id = record['action_id']
                action_stats[action_id]['attempts'] += 1
                if record['success']:
                    action_stats[action_id]['successes'] += 1
        
        # Find action with highest success rate (with minimum attempts)
        best_action = None
        best_success_rate = 0
        
        for action in available_actions:
            stats = action_stats.get(action.action_id)
            if stats and stats['attempts'] >= 3:  # Minimum confidence threshold
                success_rate = stats['successes'] / stats['attempts']
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_action = action
        
        # If no historical data, use AI to reason about the problem
        if not best_action and available_actions:
            best_action = self._ai_select_action(problem, available_actions)
        
        return best_action
    
    def _ai_select_action(self,
                         problem: Problem,
                         actions: List[RemediationAction]) -> Optional[RemediationAction]:
        """Use AI to select remediation when no historical data exists"""
        
        actions_text = "\n".join([
            f"- {a.action_id}: {a.action_type} (safety: {a.safety_level.value})"
            for a in actions
        ])
        
        prompt = f"""Select the best remediation action for this problem:

Problem Type: {problem.problem_type}
Service: {problem.service}
Severity: {problem.severity}
Symptoms: {json.dumps(problem.symptoms, indent=2)}

Available Actions:
{actions_text}

Which action would be most effective? Respond with just the action_id."""

        try:
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 100,
                    'messages': [{
                        'role': 'user',
                        'content': prompt
                    }]
                })
            )
            
            result = json.loads(response['body'].read())
            selected_id = result['content'][0]['text'].strip()
            
            # Find matching action
            for action in actions:
                if action.action_id in selected_id:
                    return action
        
        except Exception as e:
            print(f"AI selection failed: {e}")
        
        # Fallback: return safest action
        return min(actions, key=lambda a: a.safety_level.value)


class RunbookExecutor:
    """
    Executes automated runbooks with step-by-step orchestration.
    """
    
    def __init__(self):
        """Initialize runbook executor"""
        self.runbooks: Dict[str, List[RunbookStep]] = {}
    
    def register_runbook(self, problem_type: str, steps: List[RunbookStep]):
        """Register a runbook for a problem type"""
        self.runbooks[problem_type] = steps
    
    def execute_runbook(self, problem_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute runbook for a problem type.
        
        Args:
            problem_type: Type of problem
            context: Execution context
            
        Returns:
            Execution result
        """
        if problem_type not in self.runbooks:
            return {
                'success': False,
                'error': f"No runbook found for {problem_type}"
            }
        
        steps = self.runbooks[problem_type]
        executed_steps = []
        
        for i, step in enumerate(steps):
            step_result = {
                'step_id': step.step_id,
                'description': step.description,
                'started_at': datetime.now()
            }
            
            try:
                # Execute action
                action_result = step.action(context)
                step_result['action_result'] = action_result
                step_result['completed_at'] = datetime.now()
                
                # Verify if verification function provided
                if step.verification:
                    verification_result = step.verification(context)
                    step_result['verification'] = verification_result
                    
                    if not verification_result.get('passed', False):
                        step_result['success'] = False
                        
                        if not step.continue_on_failure:
                            # Rollback previous steps
                            self._rollback_steps(executed_steps, context)
                            return {
                                'success': False,
                                'error': f"Step {step.step_id} verification failed",
                                'executed_steps': executed_steps
                            }
                else:
                    step_result['success'] = True
                
            except Exception as e:
                step_result['success'] = False
                step_result['error'] = str(e)
                step_result['completed_at'] = datetime.now()
                
                if not step.continue_on_failure:
                    self._rollback_steps(executed_steps, context)
                    return {
                        'success': False,
                        'error': f"Step {step.step_id} failed: {e}",
                        'executed_steps': executed_steps
                    }
            
            executed_steps.append(step_result)
        
        return {
            'success': True,
            'executed_steps': executed_steps
        }
    
    def _rollback_steps(self, executed_steps: List[Dict], context: Dict[str, Any]):
        """Rollback executed steps in reverse order"""
        for step_result in reversed(executed_steps):
            # In production, would call actual rollback functions
            print(f"Rolling back step: {step_result['step_id']}")


class ProgressiveRemediator:
    """
    Implements progressive remediation - tries gentler actions before escalating.
    """
    
    def __init__(self, bedrock_client):
        """Initialize progressive remediator"""
        self.bedrock_client = bedrock_client
    
    def remediate_progressively(self,
                                problem: Problem,
                                actions: List[RemediationAction],
                                max_attempts: int = 3) -> RemediationExecution:
        """
        Try remediation actions progressively from gentle to aggressive.
        
        Args:
            problem: Problem to remediate
            actions: List of actions sorted by increasing impact
            max_attempts: Maximum number of attempts
            
        Returns:
            Final remediation execution result
        """
        # Sort actions by safety level (safest first)
        sorted_actions = sorted(actions, key=lambda a: a.safety_level.value)
        
        last_execution = None
        
        for i, action in enumerate(sorted_actions[:max_attempts]):
            print(f"\nAttempt {i+1}/{max_attempts}: Trying {action.action_type}")
            
            execution = RemediationExecution(
                execution_id=f"exec_{problem.problem_id}_{i}",
                problem=problem,
                action=action,
                status=RemediationStatus.EXECUTING,
                started_at=datetime.now()
            )
            
            # Simulate action execution
            success = self._execute_action(action, problem)
            
            execution.status = RemediationStatus.SUCCESS if success else RemediationStatus.FAILED
            execution.completed_at = datetime.now()
            execution.success = success
            
            # Verify remediation
            if success:
                verification = self._verify_remediation(problem)
                execution.verification_result = verification
                
                if verification['problem_resolved']:
                    print(f"✓ Problem resolved with {action.action_type}")
                    return execution
                else:
                    print(f"✗ Problem persists after {action.action_type}, escalating...")
            else:
                print(f"✗ Action {action.action_type} failed")
            
            last_execution = execution
        
        # All attempts exhausted - escalate
        print("\n⚠ All remediation attempts exhausted, escalating to humans")
        last_execution.status = RemediationStatus.ESCALATED
        
        return last_execution
    
    def _execute_action(self, action: RemediationAction, problem: Problem) -> bool:
        """
        Simulate action execution.
        
        In production, this would call actual infrastructure APIs.
        """
        # Simulate execution time
        time.sleep(0.1)
        
        # Simulate success rate based on action type and problem match
        import random
        
        # Higher success for well-matched actions
        if action.action_type == 'restart' and problem.problem_type in ['process_crash', 'memory_leak']:
            return random.random() > 0.1  # 90% success
        elif action.action_type == 'cleanup' and problem.problem_type == 'disk_full':
            return random.random() > 0.2  # 80% success
        else:
            return random.random() > 0.5  # 50% success
    
    def _verify_remediation(self, problem: Problem) -> Dict[str, Any]:
        """
        Verify that remediation resolved the problem.
        
        In production, this would check actual metrics/health.
        """
        import random
        
        # Simulate verification
        problem_resolved = random.random() > 0.3  # 70% chance resolved
        
        return {
            'problem_resolved': problem_resolved,
            'verification_time': datetime.now(),
            'metrics_improved': problem_resolved
        }


class AutomatedRemediationPipeline:
    """
    Production pipeline orchestrating all automated remediation components.
    """
    
    def __init__(self, aws_region: str = 'us-east-1'):
        """
        Initialize automated remediation pipeline.
        
        Args:
            aws_region: AWS region for Bedrock
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        self.action_library = RemediationActionLibrary()
        self.pattern_selector = PatternBasedRemediationSelector(self.bedrock_client)
        self.runbook_executor = RunbookExecutor()
        self.progressive_remediator = ProgressiveRemediator(self.bedrock_client)
        
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=300)
        self.rate_limiter = RateLimiter(max_actions_per_hour=10)
        
        self.executions: List[RemediationExecution] = []
        self.approval_queue: List[RemediationExecution] = []
        
        self._initialize_runbooks()
    
    def _initialize_runbooks(self):
        """Initialize standard runbooks"""
        
        # Runbook for process crash
        crash_runbook = [
            RunbookStep(
                step_id='check_health',
                description='Check service health',
                action=lambda ctx: {'healthy': False},
                verification=lambda ctx: {'passed': not ctx.get('healthy', True)}
            ),
            RunbookStep(
                step_id='restart_service',
                description='Restart crashed service',
                action=lambda ctx: {'restarted': True}
            ),
            RunbookStep(
                step_id='verify_restart',
                description='Verify service is healthy',
                action=lambda ctx: {'health_check': 'passed'},
                verification=lambda ctx: {'passed': True}
            )
        ]
        
        self.runbook_executor.register_runbook('process_crash', crash_runbook)
    
    def handle_problem(self,
                      problem: Problem,
                      auto_execute: bool = True) -> RemediationExecution:
        """
        Handle a detected problem with automated remediation.
        
        Args:
            problem: Detected problem
            auto_execute: Whether to auto-execute or require approval
            
        Returns:
            Remediation execution result
        """
        print(f"\n{'='*80}")
        print(f"HANDLING PROBLEM: {problem.problem_id}")
        print(f"{'='*80}")
        print(f"Type: {problem.problem_type}")
        print(f"Service: {problem.service}")
        print(f"Severity: {problem.severity}")
        print(f"Symptoms: {json.dumps(problem.symptoms, indent=2)}")
        print()
        
        # Get available actions
        available_actions = self.action_library.get_actions_for_problem(problem.problem_type)
        
        if not available_actions:
            print("⚠ No remediation actions available for this problem type")
            return self._create_escalated_execution(problem)
        
        # Select best action using pattern-based learning
        selected_action = self.pattern_selector.select_remediation(problem, available_actions)
        
        if not selected_action:
            print("⚠ Could not select appropriate remediation action")
            return self._create_escalated_execution(problem)
        
        print(f"Selected Action: {selected_action.action_type} ({selected_action.action_id})")
        print(f"Safety Level: {selected_action.safety_level.value}")
        print(f"Requires Approval: {selected_action.requires_approval}")
        print()
        
        # Check safety mechanisms
        can_execute, reason = self._check_safety(selected_action)
        
        if not can_execute:
            print(f"✗ Cannot execute: {reason}")
            return self._create_escalated_execution(problem, reason)
        
        # Check if requires approval
        if selected_action.requires_approval and not auto_execute:
            print("⚠ Action requires approval - adding to approval queue")
            execution = self._create_pending_execution(problem, selected_action)
            self.approval_queue.append(execution)
            return execution
        
        # Execute remediation
        if problem.problem_type == 'process_crash':
            # Use runbook for structured problems
            print("Executing via runbook...")
            result = self.runbook_executor.execute_runbook(problem.problem_type, {})
            execution = self._create_execution_from_runbook(problem, selected_action, result)
        else:
            # Use progressive remediation
            print("Executing progressive remediation...")
            execution = self.progressive_remediator.remediate_progressively(
                problem,
                available_actions
            )
        
        # Record result for learning
        self.pattern_selector.record_remediation_result(
            problem,
            selected_action,
            execution.success
        )
        
        # Update safety mechanisms
        if execution.success:
            self.circuit_breaker.record_success(selected_action.action_type)
        else:
            self.circuit_breaker.record_failure(selected_action.action_type)
        
        self.executions.append(execution)
        
        # Generate AI summary
        self._generate_execution_summary(execution)
        
        return execution
    
    def _check_safety(self, action: RemediationAction) -> Tuple[bool, str]:
        """Check safety mechanisms before execution"""
        
        # Check circuit breaker
        can_execute, reason = self.circuit_breaker.can_execute(action.action_type)
        if not can_execute:
            return False, reason
        
        # Check rate limiter
        can_execute, reason = self.rate_limiter.can_execute(action.action_type)
        if not can_execute:
            return False, reason
        
        return True, "Safety checks passed"
    
    def _create_escalated_execution(self,
                                   problem: Problem,
                                   reason: str = "No available remediation") -> RemediationExecution:
        """Create execution record for escalated problem"""
        return RemediationExecution(
            execution_id=f"escalated_{problem.problem_id}",
            problem=problem,
            action=RemediationAction(
                action_id='escalate',
                action_type='escalate',
                target='human',
                parameters={},
                safety_level=RemediationSafety.SAFE,
                estimated_duration_seconds=0
            ),
            status=RemediationStatus.ESCALATED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            error_message=reason
        )
    
    def _create_pending_execution(self,
                                 problem: Problem,
                                 action: RemediationAction) -> RemediationExecution:
        """Create execution record for pending approval"""
        return RemediationExecution(
            execution_id=f"pending_{problem.problem_id}",
            problem=problem,
            action=action,
            status=RemediationStatus.PENDING,
            started_at=datetime.now()
        )
    
    def _create_execution_from_runbook(self,
                                      problem: Problem,
                                      action: RemediationAction,
                                      runbook_result: Dict) -> RemediationExecution:
        """Create execution from runbook result"""
        return RemediationExecution(
            execution_id=f"exec_{problem.problem_id}",
            problem=problem,
            action=action,
            status=RemediationStatus.SUCCESS if runbook_result['success'] else RemediationStatus.FAILED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            success=runbook_result['success'],
            error_message=runbook_result.get('error')
        )
    
    def _generate_execution_summary(self, execution: RemediationExecution):
        """Generate AI summary of remediation execution"""
        
        prompt = f"""Summarize this automated remediation execution:

Problem: {execution.problem.problem_type}
Service: {execution.problem.service}
Action Taken: {execution.action.action_type}
Status: {execution.status.value}
Success: {execution.success}

Provide a brief 2-sentence summary suitable for an incident notification."""

        try:
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 150,
                    'messages': [{
                        'role': 'user',
                        'content': prompt
                    }]
                })
            )
            
            result = json.loads(response['body'].read())
            summary = result['content'][0]['text']
            
            print(f"\n📋 EXECUTION SUMMARY")
            print(f"   {summary}")
            
        except Exception as e:
            print(f"Could not generate summary: {e}")
    
    def get_remediation_statistics(self) -> Dict[str, Any]:
        """Get statistics on remediation effectiveness"""
        if not self.executions:
            return {'no_data': True}
        
        total = len(self.executions)
        successful = len([e for e in self.executions if e.success])
        failed = len([e for e in self.executions if not e.success and e.status != RemediationStatus.ESCALATED])
        escalated = len([e for e in self.executions if e.status == RemediationStatus.ESCALATED])
        
        # Success by action type
        by_action = defaultdict(lambda: {'total': 0, 'success': 0})
        for execution in self.executions:
            action_type = execution.action.action_type
            by_action[action_type]['total'] += 1
            if execution.success:
                by_action[action_type]['success'] += 1
        
        return {
            'total_remediations': total,
            'successful': successful,
            'failed': failed,
            'escalated': escalated,
            'success_rate': successful / total if total > 0 else 0,
            'by_action_type': dict(by_action)
        }


# Example usage and realistic scenarios
def generate_sample_problems() -> List[Problem]:
    """Generate sample problems for demonstration"""
    problems = []
    
    # Process crash
    problems.append(Problem(
        problem_id='prob_001',
        timestamp=datetime.now(),
        service='payment-service',
        problem_type='process_crash',
        severity='critical',
        symptoms={
            'exit_code': 137,
            'last_log': 'OutOfMemoryError',
            'uptime_seconds': 3600
        },
        detected_by='kubernetes'
    ))
    
    # Memory leak
    problems.append(Problem(
        problem_id='prob_002',
        timestamp=datetime.now(),
        service='user-service',
        problem_type='memory_leak',
        severity='high',
        symptoms={
            'memory_usage_percent': 92,
            'memory_growth_rate_mb_per_hour': 50,
            'gc_frequency': 'increasing'
        },
        detected_by='prometheus'
    ))
    
    # Disk full
    problems.append(Problem(
        problem_id='prob_003',
        timestamp=datetime.now(),
        service='logging-service',
        problem_type='disk_full',
        severity='high',
        symptoms={
            'disk_usage_percent': 96,
            'largest_dir': '/var/log',
            'growth_rate_gb_per_day': 10
        },
        detected_by='monitoring'
    ))
    
    # High latency
    problems.append(Problem(
        problem_id='prob_004',
        timestamp=datetime.now(),
        service='api-gateway',
        problem_type='high_latency',
        severity='medium',
        symptoms={
            'p95_latency_ms': 2500,
            'baseline_p95_ms': 200,
            'affected_endpoints': ['/api/search', '/api/checkout']
        },
        detected_by='apm'
    ))
    
    return problems


def main():
    """
    Demonstrate automated remediation pipeline with realistic scenarios.
    """
    print("=" * 80)
    print("Chapter 15: Automated Remediation")
    print("AI-Powered Observability: From Noise to Insight")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("Initializing automated remediation pipeline...")
    pipeline = AutomatedRemediationPipeline(aws_region='us-east-1')
    print("✓ Pipeline initialized with safety mechanisms")
    print(f"  - Circuit breaker: {pipeline.circuit_breaker.failure_threshold} failure threshold")
    print(f"  - Rate limiter: {pipeline.rate_limiter.max_actions_per_hour} actions/hour max")
    print()
    
    # Generate sample problems
    problems = generate_sample_problems()
    
    # Handle each problem
    for problem in problems:
        execution = pipeline.handle_problem(problem, auto_execute=True)
        
        print(f"\n{'─'*80}")
        print(f"RESULT: {execution.status.value.upper()}")
        if execution.success:
            print(f"✓ Problem successfully remediated")
        elif execution.status == RemediationStatus.ESCALATED:
            print(f"⚠ Problem escalated to humans")
        else:
            print(f"✗ Remediation failed")
        print()
    
    # Show statistics
    print("=" * 80)
    print("REMEDIATION STATISTICS")
    print("=" * 80)
    print()
    
    stats = pipeline.get_remediation_statistics()
    
    print(f"Total Remediations: {stats['total_remediations']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Escalated: {stats['escalated']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    
    print(f"\nBy Action Type:")
    for action_type, action_stats in stats['by_action_type'].items():
        success_rate = action_stats['success'] / action_stats['total'] if action_stats['total'] > 0 else 0
        print(f"  {action_type}: {action_stats['success']}/{action_stats['total']} ({success_rate:.0%})")
    
    print("\n" + "=" * 80)
    print("Automated Remediation Demonstration Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
