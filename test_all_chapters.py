"""
Comprehensive tests for all chapter code examples.
Uses the default AWS CLI profile for Bedrock client where needed.
"""

import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import boto3

PASS = 0
FAIL = 0
ERRORS = []

# Shared Bedrock client using default AWS CLI profile
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")


def test(name, func):
    global PASS, FAIL, ERRORS
    try:
        func()
        PASS += 1
        print(f"  PASS: {name}")
    except Exception as e:
        FAIL += 1
        ERRORS.append((name, e))
        print(f"  FAIL: {name}")
        traceback.print_exc()
        print()


# ============================================================
# Chapter 3: Building Blocks
# ============================================================
print("\n=== Chapter 3: Building Blocks ===")

def test_ch3_anomaly_detector_iforest():
    from chapter_03_building_blocks import AnomalyDetector
    data = np.concatenate([np.random.normal(50, 5, 100), [200, -50]])
    indices, scores = AnomalyDetector.isolation_forest_detection(data, contamination=0.05)
    assert len(indices) > 0, "Should detect at least one anomaly"
    assert len(scores) == len(data), "Should return score for each point"

def test_ch3_zscore():
    from chapter_03_building_blocks import AnomalyDetector
    data = np.array([10.0, 11, 12, 10, 11, 50, 10, 11])
    indices, scores = AnomalyDetector.zscore_detection(data, threshold=2.0)
    assert 5 in indices, "Should detect the outlier at index 5"

def test_ch3_iqr():
    from chapter_03_building_blocks import AnomalyDetector
    data = np.concatenate([np.random.normal(50, 5, 100), [200]])
    indices, scores = AnomalyDetector.iqr_detection(data, multiplier=1.5)
    assert len(indices) > 0, "Should detect outlier"

test("Isolation Forest detection", test_ch3_anomaly_detector_iforest)
test("Z-score detection", test_ch3_zscore)
test("IQR detection", test_ch3_iqr)


# ============================================================
# Chapter 5: Pattern Recognition
# ============================================================
print("\n=== Chapter 5: Pattern Recognition ===")

def test_ch5_drain_parser():
    from chapter_05_pattern_recognition import DrainParser
    parser = DrainParser(depth=4, similarity_threshold=0.4)
    # Parse similar log messages — Drain groups by token count + prefix
    t1_id, t1 = parser.parse("User john logged in from 192.168.1.1 successfully")
    t2_id, t2 = parser.parse("User jane logged in from 10.0.0.5 successfully")
    t3_id, t3 = parser.parse("Connection timeout to database server db01")
    # Verify templates were created
    assert len(parser.templates) >= 2, f"Should have at least 2 templates, got {len(parser.templates)}"
    # t1 and t2 should either share a template or produce templates with wildcards
    assert t3_id != t1_id, "Different log patterns should get different templates"

def test_ch5_cluster_evolution():
    from chapter_05_pattern_recognition import ClusterEvolutionTracker, SemanticCluster
    tracker = ClusterEvolutionTracker()
    # Create SemanticCluster objects for snapshots
    snap1_clusters = {
        0: SemanticCluster(cluster_id=0, centroid=np.zeros(8), log_indices=[0,1,2], label="A"),
        1: SemanticCluster(cluster_id=1, centroid=np.ones(8), log_indices=[3,4], label="B"),
    }
    snap2_clusters = {
        0: SemanticCluster(cluster_id=0, centroid=np.zeros(8), log_indices=list(range(10)), label="A"),
        1: SemanticCluster(cluster_id=1, centroid=np.ones(8), log_indices=[10,11], label="B"),
        2: SemanticCluster(cluster_id=2, centroid=np.ones(8)*2, log_indices=[12,13,14], label="C"),
    }
    tracker.add_snapshot(datetime.now() - timedelta(hours=1), snap1_clusters, total_logs=5)
    tracker.add_snapshot(datetime.now(), snap2_clusters, total_logs=15)
    changes = tracker.detect_changes(growth_threshold=0.5)
    types = [c['type'] for c in changes]
    assert 'emerging' in types, f"Should detect new cluster C, got types: {types}"

test("Drain parser template extraction", test_ch5_drain_parser)
test("Cluster evolution tracking", test_ch5_cluster_evolution)


# ============================================================
# Chapter 7: Statistical vs ML
# ============================================================
print("\n=== Chapter 7: Statistical vs ML ===")

def test_ch7_zscore_detector():
    from chapter_07_statistical_vs_ml import ZScoreDetector
    detector = ZScoreDetector(threshold=2.5)
    normal = np.random.normal(100, 10, 200)
    detector.fit(normal)
    test_data = np.array([100, 105, 95, 200, 5])
    predictions = detector.predict(test_data)
    assert predictions[3] == 1, "Should flag 200 as anomaly"
    assert predictions[0] == 0, "100 should be normal"

def test_ch7_iqr_detector():
    from chapter_07_statistical_vs_ml import IQRDetector
    detector = IQRDetector(k=1.5)
    data = np.concatenate([np.random.normal(50, 5, 100), [200]])
    detector.fit(data)
    predictions = detector.predict(np.array([50, 200]))
    assert predictions[1] == 1, "Should flag 200 as anomaly"

def test_ch7_isolation_forest_detector():
    from chapter_07_statistical_vs_ml import IsolationForestDetector
    detector = IsolationForestDetector(contamination=0.05)
    normal = np.random.normal(0, 1, (200, 2))
    detector.fit(normal)
    test_pts = np.array([[0, 0], [10, 10]])
    predictions = detector.predict(test_pts)
    assert predictions[1] == 1, "Should flag far-out point as anomaly"

def test_ch7_tiered_detector():
    from chapter_07_statistical_vs_ml import TieredDetector, ZScoreDetector, IsolationForestDetector
    fast = ZScoreDetector(threshold=2.0)
    precise = IsolationForestDetector(contamination=0.1)
    tiered = TieredDetector(fast, precise)
    data = np.random.normal(50, 5, 200)
    tiered.fit(data)
    test_data = np.concatenate([np.random.normal(50, 5, 50), [200, 300]])
    predictions = tiered.predict(test_data)
    assert len(predictions) == len(test_data)

def test_ch7_threshold_optimizer():
    from chapter_07_statistical_vs_ml import ThresholdOptimizer
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.15, 0.25])
    labels = np.array([0, 0, 0, 1, 1, 0, 0])
    fpr_threshold = ThresholdOptimizer.optimize_fpr(scores, labels, target_fpr=0.1)
    assert isinstance(fpr_threshold, float)
    recall_threshold = ThresholdOptimizer.optimize_recall(scores, labels, target_recall=0.95)
    assert isinstance(recall_threshold, float)

test("Z-score detector", test_ch7_zscore_detector)
test("IQR detector", test_ch7_iqr_detector)
test("Isolation Forest detector", test_ch7_isolation_forest_detector)
test("Tiered detector", test_ch7_tiered_detector)
test("Threshold optimizer", test_ch7_threshold_optimizer)


# ============================================================
# Chapter 8: Time Series Anomaly Detection
# ============================================================
print("\n=== Chapter 8: Time Series Anomaly Detection ===")

def test_ch8_moving_average_detector():
    from chapter_08_time_series_anomaly_detection import MovingAverageDetector
    detector = MovingAverageDetector(window_size=10, threshold_std=2.5)
    timestamps = [datetime.now() + timedelta(minutes=i) for i in range(100)]
    values = list(np.random.normal(100, 5, 95)) + [200, 210, 195, 205, 190]
    df = pd.DataFrame({'timestamp': timestamps, 'value': values})
    anomalies = detector.detect_batch(df)
    assert len(anomalies) > 0, "Should detect anomalies in spike region"

def test_ch8_isolation_forest_ts():
    from chapter_08_time_series_anomaly_detection import IsolationForestTimeSeriesDetector
    detector = IsolationForestTimeSeriesDetector(window_size=5, contamination=0.05)
    timestamps = [datetime.now() + timedelta(minutes=i) for i in range(200)]
    values = list(np.random.normal(100, 5, 195)) + [300, 310, 290, 305, 295]
    df = pd.DataFrame({'timestamp': timestamps, 'value': values})
    # Must fit before detect
    detector.fit(df)
    anomalies = detector.detect_batch(df)
    assert isinstance(anomalies, list), "Should return list of anomalies"

def test_ch8_stl_detector():
    from chapter_08_time_series_anomaly_detection import STLDecompositionDetector, STATSMODELS_AVAILABLE
    if not STATSMODELS_AVAILABLE:
        print("    (skipped: statsmodels not available)")
        return
    # seasonal_period must be odd for STL
    detector = STLDecompositionDetector(seasonal_period=25, threshold_std=3.0)
    n = 250  # 10 full cycles
    timestamps = pd.date_range(start='2026-01-01', periods=n, freq='h')
    seasonal = [10 * np.sin(2 * np.pi * i / 25) for i in range(n)]
    values = [100 + s + np.random.normal(0, 2) for s in seasonal]
    values[125] = 250  # big anomaly mid-series
    df = pd.DataFrame({'timestamp': timestamps, 'value': values})
    df.index = timestamps
    anomalies = detector.detect_batch(df)
    assert len(anomalies) > 0, "Should detect the injected anomaly"

test("Moving average detector", test_ch8_moving_average_detector)
test("Isolation Forest time series detector", test_ch8_isolation_forest_ts)
test("STL decomposition detector", test_ch8_stl_detector)


# ============================================================
# Chapter 9: Behavioral Anomaly Detection
# ============================================================
print("\n=== Chapter 9: Behavioral Anomaly Detection ===")

def test_ch9_user_profile():
    from chapter_09_behavioral_anomaly_detection import UserBehaviorAnalyzer, UserEvent
    analyzer = UserBehaviorAnalyzer(bedrock_client=bedrock_client)
    events = []
    for i in range(50):
        events.append(UserEvent(
            user_id="user1",
            timestamp=datetime.now() - timedelta(hours=50-i),
            action="login",
            resource="/api/data",
            source_ip="10.0.0.1",
            success=True,
            metadata={}
        ))
    profile = analyzer.build_user_profile(events, "user1")
    assert profile.user_id == "user1"
    assert profile.activity_baseline > 0
    assert len(profile.typical_actions) > 0

def test_ch9_graph_detector():
    from chapter_09_behavioral_anomaly_detection import GraphAnomalyDetector
    detector = GraphAnomalyDetector(bedrock_client=bedrock_client)
    # Build historical graph
    historical = {
        'service-a': {'service-b': 100, 'service-c': 50},
        'service-b': {'service-c': 80},
    }
    detector.graph = historical
    # Recent graph with new edge
    recent = {
        'service-a': {'service-b': 100, 'service-c': 50, 'service-d': 5},
        'service-b': {'service-c': 80},
    }
    anomaly = detector._detect_new_edges(recent)
    assert anomaly is not None, "Should detect new edge service-a -> service-d"

test("User behavior profiling", test_ch9_user_profile)
test("Graph anomaly detection (new edges)", test_ch9_graph_detector)


# ============================================================
# Chapter 10: Alert Fatigue
# ============================================================
print("\n=== Chapter 10: Alert Fatigue ===")

def test_ch10_fatigue_score():
    from chapter_10_alert_fatigue_problem import AlertFatigueAnalyzer, AlertFatigueMetrics
    analyzer = AlertFatigueAnalyzer(bedrock_client=bedrock_client)
    metrics = AlertFatigueMetrics(
        period_start=datetime.now() - timedelta(days=7),
        period_end=datetime.now(),
        total_alerts=1000,
        actionable_alerts=200,
        false_positives=700,
        signal_to_noise_ratio=0.2,
        average_response_time_minutes=15.0,
        average_resolution_time_minutes=45.0,
        night_pages=30,
        repeat_alerts=300,
        unique_alert_types=25,
        top_noisy_alerts=[("high_cpu", 200), ("disk_warn", 150)],
        severity_distribution={"critical": 50, "warning": 400, "info": 550},
        fatigue_score=0.0  # will be calculated
    )
    score = analyzer.calculate_fatigue_score(metrics)
    assert 0 <= score <= 100, f"Fatigue score should be 0-100, got {score}"
    assert score > 50, f"High noise ratio should produce high fatigue score, got {score}"

test("Alert fatigue score calculation", test_ch10_fatigue_score)


# ============================================================
# Chapter 11: Correlation and Deduplication
# ============================================================
print("\n=== Chapter 11: Correlation & Deduplication ===")

def test_ch11_deduplication():
    from chapter_11_correlation_deduplication import AlertDeduplicator, Alert
    dedup = AlertDeduplicator()
    now = datetime.now()
    a1 = Alert(alert_id="a1", timestamp=now, source="prometheus",
               severity="warning", title="High CPU on web-01",
               message="CPU at 95%", affected_entity="web-01", tags=["cpu"])
    a2 = Alert(alert_id="a2", timestamp=now + timedelta(seconds=30),
               source="prometheus", severity="warning",
               title="High CPU on web-01", message="CPU at 96%",
               affected_entity="web-01", tags=["cpu"])
    a3 = Alert(alert_id="a3", timestamp=now + timedelta(seconds=60),
               source="prometheus", severity="critical",
               title="Disk full on db-01", message="Disk at 99%",
               affected_entity="db-01", tags=["disk"])
    # exact_deduplicate returns (is_duplicate, existing_alert_id)
    is_dup1, _ = dedup.exact_deduplicate(a1)
    is_dup2, _ = dedup.exact_deduplicate(a2)
    is_dup3, _ = dedup.exact_deduplicate(a3)
    assert is_dup1 is False, "First alert should not be a duplicate"
    assert is_dup2 is True, "Same alert should be a duplicate"
    assert is_dup3 is False, "Different alert should not be a duplicate"

test("Alert deduplication", test_ch11_deduplication)


# ============================================================
# Chapter 12: Predictive Alerting
# ============================================================
print("\n=== Chapter 12: Predictive Alerting ===")

def test_ch12_resource_exhaustion():
    from chapter_12_predictive_alerting import ResourceExhaustionPredictor, MetricDataPoint
    predictor = ResourceExhaustionPredictor(bedrock_client=bedrock_client)
    # Monkey-patch LLM calls to avoid Bedrock
    predictor._generate_prediction_explanation = lambda *a, **kw: "Test prediction"
    predictor._generate_recommendations = lambda *a, **kw: ["Scale disk"]
    now = datetime.now()
    for i in range(48):
        dp = MetricDataPoint(
            timestamp=now - timedelta(hours=48-i),
            value=50 + (35 * i / 48),
            entity="db-01",
            metric_name="disk_usage_pct"
        )
        predictor.add_metric_data(dp)
    alert = predictor.predict_resource_exhaustion(
        "db-01", "disk_usage_pct", threshold=95.0
    )
    assert alert is not None, "Should predict disk exhaustion"
    assert alert.severity in ['critical', 'warning', 'info']

test("Resource exhaustion prediction", test_ch12_resource_exhaustion)


# ============================================================
# Chapter 13: Performance Analysis
# ============================================================
print("\n=== Chapter 13: Performance Analysis ===")

def test_ch13_regression_detection():
    from chapter_13_ai_performance_analysis import PerformanceRegressionDetector
    detector = PerformanceRegressionDetector(bedrock_client=bedrock_client)
    # Monkey-patch the LLM description method to avoid Bedrock call
    detector._generate_regression_description = lambda *args, **kwargs: "Test regression detected"
    np.random.seed(42)
    before = list(np.random.normal(100, 10, 50))
    after = list(np.random.normal(150, 10, 50))
    result = detector.detect_regression(
        before, after, "latency_ms", "api-service", "deploy-123"
    )
    assert result is not None, "Should detect 50% regression"
    assert result.severity in ['critical', 'warning']

def test_ch13_n_plus_1():
    from chapter_13_ai_performance_analysis import TraceAnalyzer, DistributedTrace, TraceSpan
    analyzer = TraceAnalyzer(bedrock_client=bedrock_client)
    now = datetime.now()
    spans = []
    for i in range(10):
        spans.append(TraceSpan(
            trace_id="t1", span_id=f"s{i}",
            parent_span_id=None,
            service="db-service", operation="SELECT",
            start_time=now + timedelta(milliseconds=i*10),
            duration_ms=5.0
        ))
    trace = DistributedTrace(
        trace_id="t1", spans=spans,
        total_duration_ms=100.0, is_slow=True,
        endpoint="/api/orders", timestamp=now
    )
    insight = analyzer._detect_n_plus_1(trace)
    assert insight is not None, "Should detect N+1 pattern"
    assert "N+1" in insight.title

test("Performance regression detection", test_ch13_regression_detection)
test("N+1 query pattern detection", test_ch13_n_plus_1)


# ============================================================
# Chapter 14: Capacity Planning
# ============================================================
print("\n=== Chapter 14: Capacity Planning ===")

def test_ch14_scaling_recommendation():
    from chapter_14_capacity_planning import ScalingOptimizer
    optimizer = ScalingOptimizer(bedrock_client=bedrock_client)
    # Monkey-patch the LLM rationale method to avoid Bedrock call
    optimizer._generate_scaling_rationale = lambda *args, **kwargs: "Test scaling rationale"
    forecast = {
        'peak_demand': 800,
        'current_demand': 500,
        'peak_time': datetime.now() + timedelta(hours=12),
        'confidence': 0.85
    }
    rec = optimizer.generate_scaling_recommendation(
        service="api-server",
        demand_forecast=forecast,
        current_capacity=700,
        target_utilization=70.0
    )
    assert rec.action == 'scale_up', f"Should recommend scale_up, got {rec.action}"
    assert rec.recommended_value > rec.current_value

test("Scaling recommendation generation", test_ch14_scaling_recommendation)


# ============================================================
# Chapter 15: Automated Remediation
# ============================================================
print("\n=== Chapter 15: Automated Remediation ===")

def test_ch15_circuit_breaker():
    from chapter_15_automated_remediation import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=3, timeout_seconds=60)
    can, reason = cb.can_execute("restart")
    assert can is True, "Should be closed initially"
    cb.record_failure("restart")
    cb.record_failure("restart")
    cb.record_failure("restart")
    can, reason = cb.can_execute("restart")
    assert can is False, "Should be open after 3 failures"
    assert "Circuit open" in reason or "circuit" in reason.lower()

def test_ch15_remediation_selector():
    from chapter_15_automated_remediation import (
        PatternBasedRemediationSelector, Problem, RemediationAction, RemediationSafety
    )
    selector = PatternBasedRemediationSelector(bedrock_client=bedrock_client)
    for i in range(5):
        selector.pattern_history.append({
            'problem_type': 'high_cpu',
            'action_id': 'restart_service',
            'success': True
        })
    for i in range(5):
        selector.pattern_history.append({
            'problem_type': 'high_cpu',
            'action_id': 'scale_up',
            'success': i < 2
        })
    problem = Problem(
        problem_id="p1", timestamp=datetime.now(),
        service="web-01", problem_type="high_cpu",
        severity="warning", symptoms={"cpu": 95},
        detected_by="prometheus"
    )
    actions = [
        RemediationAction(
            action_id='restart_service', action_type='restart',
            target='web-01', parameters={},
            safety_level=RemediationSafety.SAFE,
            estimated_duration_seconds=30
        ),
        RemediationAction(
            action_id='scale_up', action_type='scale',
            target='web-01', parameters={},
            safety_level=RemediationSafety.MODERATE,
            estimated_duration_seconds=120
        ),
    ]
    selected = selector.select_remediation(problem, actions)
    assert selected is not None
    assert selected.action_id == 'restart_service', "Should pick highest success rate"

test("Circuit breaker", test_ch15_circuit_breaker)
test("Pattern-based remediation selection", test_ch15_remediation_selector)


# ============================================================
# Chapter 16: Threat Detection
# ============================================================
print("\n=== Chapter 16: Threat Detection ===")

def test_ch16_beaconing_detection():
    from chapter_16_threat_detection import NetworkAnomalyDetector, NetworkEvent
    detector = NetworkAnomalyDetector(bedrock_client=bedrock_client)
    now = datetime.now()
    events = []
    for i in range(10):
        events.append(NetworkEvent(
            timestamp=now + timedelta(seconds=i*60),
            source_ip="10.0.0.50",
            destination_ip="evil.example.com",
            source_port=44000 + i,
            destination_port=443,
            protocol="TCP",
            bytes_sent=100,
            bytes_received=200,
            duration_seconds=1.0,
            country_code="US",
            is_internal=False
        ))
    alert = detector._detect_beaconing(events, "10.0.0.50")
    assert alert is not None, "Should detect regular beaconing"
    assert alert.threat_type == "c2_beaconing"

def test_ch16_kill_chain():
    from chapter_16_threat_detection import (
        KillChainDetector, ThreatAlert, ThreatSeverity, AttackStage
    )
    detector = KillChainDetector(bedrock_client=bedrock_client)
    now = datetime.now()
    stages = [
        AttackStage.RECONNAISSANCE, AttackStage.INITIAL_ACCESS,
        AttackStage.LATERAL_MOVEMENT, AttackStage.EXFILTRATION
    ]
    for i, stage in enumerate(stages):
        alert = ThreatAlert(
            alert_id=f"alert_{i}",
            timestamp=now + timedelta(minutes=i*30),
            threat_type="test", severity=ThreatSeverity.HIGH,
            attack_stage=stage, source="10.0.0.50", target="various",
            description=f"Stage {stage.value}", evidence={},
            indicators=["10.0.0.50"], recommended_actions=[],
            confidence=0.8, mitre_techniques=[]
        )
        detector.add_alert(alert)
    kill_chains = detector.detect_kill_chain()
    assert len(kill_chains) > 0, "Should detect multi-stage attack"
    assert kill_chains[0].severity == ThreatSeverity.CRITICAL

test("C2 beaconing detection", test_ch16_beaconing_detection)
test("Kill chain detection", test_ch16_kill_chain)


# ============================================================
# Chapter 17: Security Log Analysis
# ============================================================
print("\n=== Chapter 17: Security Log Analysis ===")

def test_ch17_template_extractor():
    from chapter_17_security_log_analysis import LogTemplateExtractor
    extractor = LogTemplateExtractor()
    t1 = extractor.extract_template(
        "Failed login for user john from 192.168.1.100"
    )
    t2 = extractor.extract_template(
        "Failed login for user jane from 10.0.0.5"
    )
    # Both should produce same structural template
    assert t1 == t2, f"Similar logs should produce same template:\n  '{t1}'\n  '{t2}'"

def test_ch17_rare_templates():
    from chapter_17_security_log_analysis import LogTemplateExtractor
    extractor = LogTemplateExtractor()
    # Generate many of one pattern
    for i in range(20):
        extractor.extract_template(f"User user{i} logged in from 10.0.0.{i}")
    # One rare pattern
    extractor.extract_template("Kernel panic: fatal exception in interrupt")
    rare = extractor.get_rare_templates(threshold=5)
    assert len(rare) > 0, "Should find at least one rare template"

test("Log template extraction", test_ch17_template_extractor)
test("Rare template detection", test_ch17_rare_templates)


# ============================================================
# Chapter 18: UEBA
# ============================================================
print("\n=== Chapter 18: UEBA ===")

def test_ch18_impossible_travel():
    from chapter_18_ueba import (
        BehaviorAnomalyDetector, BehavioralBaselineBuilder,
        UserActivity, EntityType
    )
    baseline_builder = BehavioralBaselineBuilder()
    detector = BehaviorAnomalyDetector(
        bedrock_client=bedrock_client,
        baseline_builder=baseline_builder
    )
    now = datetime.now()
    activity1 = UserActivity(
        timestamp=now - timedelta(hours=1),
        entity_id="user1", entity_type=EntityType.USER,
        action="login", target_resource="/app",
        source_ip="1.2.3.4",
        location="New York", latitude=40.7128, longitude=-74.0060,
        bytes_transferred=0, session_id="s1", device_id="d1",
        success=True
    )
    # Directly populate recent_activities to set up state
    detector.recent_activities["user1"] = [activity1]
    activity2 = UserActivity(
        timestamp=now,
        entity_id="user1", entity_type=EntityType.USER,
        action="login", target_resource="/app",
        source_ip="5.6.7.8",
        location="Tokyo", latitude=35.6762, longitude=139.6503,
        bytes_transferred=0, session_id="s2", device_id="d2",
        success=True
    )
    anomaly = detector._check_impossible_travel(activity2)
    assert anomaly is not None, "Should detect impossible travel NY->Tokyo in 1hr"
    assert anomaly.risk_score >= 0.9

def test_ch18_risk_scorer():
    from chapter_18_ueba import (
        EntityRiskScorer, BehaviorAnomaly, AnomalyType, RiskLevel
    )
    scorer = EntityRiskScorer(bedrock_client=bedrock_client)
    now = datetime.now()
    anomaly_types = [AnomalyType.TEMPORAL, AnomalyType.ACCESS, AnomalyType.VOLUME]
    for i, atype in enumerate(anomaly_types):
        anomaly = BehaviorAnomaly(
            anomaly_id=f"a{i}",
            timestamp=now - timedelta(hours=i),
            entity_id="user1", entity_type="user",
            anomaly_type=atype,
            description="test anomaly",
            risk_score=0.7, risk_level=RiskLevel.HIGH,
            evidence={}, baseline_comparison={},
            recommended_actions=[],
            mitre_techniques=[]
        )
        scorer.add_anomaly(anomaly)
    risk = scorer.calculate_risk_score("user1")
    assert risk.overall_score > 0.3, f"Multiple anomalies should elevate risk, got {risk.overall_score}"

test("Impossible travel detection", test_ch18_impossible_travel)
test("Entity risk scoring", test_ch18_risk_scorer)


# ============================================================
# Chapter 19: Incident Response
# ============================================================
print("\n=== Chapter 19: Incident Response ===")

def test_ch19_safety_controller():
    from chapter_19_incident_response import (
        SafetyController, ResponseAction,
        ActionCategory, AutomationLevel, ActionStatus
    )
    controller = SafetyController()
    action = ResponseAction(
        action_id="a1", incident_id="inc1",
        timestamp=datetime.now(),
        action_type="isolate_host",
        category=ActionCategory.CONTAINMENT,
        target="web-01",
        parameters={},
        automation_level=AutomationLevel.BOUNDED,
        status=ActionStatus.PENDING,
        requires_approval=False
    )
    can, reason = controller.can_execute(action)
    assert can is True, f"Should pass safety checks: {reason}"
    controller.global_kill_switch = True
    can, reason = controller.can_execute(action)
    assert can is False, "Should block when kill switch is active"
    assert "kill switch" in reason.lower()

test("Safety controller checks", test_ch19_safety_controller)


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print(f"{'='*60}")

if ERRORS:
    print("\nFailed tests:")
    for name, err in ERRORS:
        print(f"  - {name}: {type(err).__name__}: {err}")

sys.exit(1 if FAIL > 0 else 0)
