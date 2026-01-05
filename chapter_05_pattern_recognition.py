"""
Chapter 5: Pattern Recognition at Scale
=========================================

Code examples demonstrating large-scale log pattern recognition:
- Template extraction (log parsing)
- Semantic clustering with embeddings
- Cluster evolution tracking
- Trend detection and analysis
- Correlation discovery
- Pattern anomaly detection
- Dimensionality reduction for visualization
- Building a pattern library

Prerequisites:
    pip install boto3 numpy scikit-learn scipy

AWS Configuration:
    Ensure you have AWS credentials configured with access to Amazon Bedrock.
    Enable: amazon.titan-embed-text-v2:0

This chapter builds on Chapter 3-4 concepts to work with patterns
across millions of log messages.
"""

import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import hashlib
import boto3
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"


def get_bedrock_client(region_name: str = "us-east-1"):
    """Create a Bedrock runtime client."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name
    )


# =============================================================================
# LARGE SAMPLE DATASET - Simulating scale patterns
# =============================================================================

def generate_sample_logs(count: int = 500) -> List[Dict]:
    """
    Generate a realistic log dataset with patterns.
    
    Includes:
    - High-volume templates (login, API requests)
    - Error patterns with variations
    - Periodic patterns (scheduled jobs)
    - Gradual trends (increasing errors)
    - Rare/novel events
    """
    import random
    
    logs = []
    base_time = datetime(2024, 1, 15, 0, 0, 0)
    
    # Template definitions with relative frequencies
    templates = [
        # High-volume normal operations
        {
            "weight": 30,
            "level": "INFO",
            "service": "api-gateway",
            "template": "Request completed: {method} {endpoint} - {status} in {duration}ms",
            "params": lambda: {
                "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
                "endpoint": random.choice(["/api/users", "/api/orders", "/api/products", "/api/payments"]),
                "status": random.choices([200, 201, 204, 400, 404, 500], weights=[70, 10, 5, 8, 5, 2])[0],
                "duration": random.randint(10, 500)
            }
        },
        {
            "weight": 20,
            "level": "INFO",
            "service": "auth-service",
            "template": "User {user} logged in from {ip}",
            "params": lambda: {
                "user": f"{random.choice(['alice', 'bob', 'carol', 'dave', 'eve'])}@example.com",
                "ip": f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}"
            }
        },
        # Database operations
        {
            "weight": 15,
            "level": "DEBUG",
            "service": "order-service",
            "template": "Query executed in {duration}ms: {query_type} on {table}",
            "params": lambda: {
                "duration": random.randint(1, 100),
                "query_type": random.choice(["SELECT", "INSERT", "UPDATE"]),
                "table": random.choice(["orders", "order_items", "customers"])
            }
        },
        # Errors - connection issues
        {
            "weight": 5,
            "level": "ERROR",
            "service": "order-service",
            "template": "Connection to {host}:{port} failed: {reason}",
            "params": lambda: {
                "host": random.choice(["db-primary", "db-replica", "cache-server"]),
                "port": random.choice([5432, 6379, 3306]),
                "reason": random.choice(["connection refused", "timeout after 5000ms", "connection reset"])
            }
        },
        # Errors - authentication
        {
            "weight": 4,
            "level": "ERROR",
            "service": "auth-service",
            "template": "Authentication failed for {user}: {reason}",
            "params": lambda: {
                "user": f"user{random.randint(1, 1000)}@example.com",
                "reason": random.choice(["invalid password", "account locked", "token expired"])
            }
        },
        # Warnings - performance
        {
            "weight": 6,
            "level": "WARN",
            "service": "api-gateway",
            "template": "Slow response: {endpoint} took {duration}ms (threshold: {threshold}ms)",
            "params": lambda: {
                "endpoint": random.choice(["/api/search", "/api/reports", "/api/analytics"]),
                "duration": random.randint(1000, 5000),
                "threshold": 500
            }
        },
        # Periodic - batch jobs (every ~50 logs to simulate hourly)
        {
            "weight": 2,
            "level": "INFO",
            "service": "batch-processor",
            "template": "Scheduled job {job_name} completed: processed {count} records in {duration}s",
            "params": lambda: {
                "job_name": random.choice(["data-sync", "report-generation", "cleanup"]),
                "count": random.randint(1000, 50000),
                "duration": random.randint(30, 300)
            }
        },
        # Memory warnings
        {
            "weight": 3,
            "level": "WARN",
            "service": "order-service",
            "template": "Memory usage high: {used}MB / {total}MB ({percent}%)",
            "params": lambda: {
                "used": random.randint(7000, 7900),
                "total": 8192,
                "percent": random.randint(85, 97)
            }
        },
        # Rare/novel events
        {
            "weight": 1,
            "level": "ERROR",
            "service": "payment-service",
            "template": "Unexpected error in {component}: {error_type}",
            "params": lambda: {
                "component": random.choice(["PaymentProcessor", "RefundHandler", "FraudDetector"]),
                "error_type": random.choice([
                    "NullPointerException",
                    "IllegalStateException", 
                    "ConcurrentModificationException",
                    "OutOfMemoryError"
                ])
            }
        },
        # Cache operations
        {
            "weight": 10,
            "level": "DEBUG",
            "service": "cache-service",
            "template": "Cache {operation}: key={key} {result}",
            "params": lambda: {
                "operation": random.choice(["GET", "SET", "DELETE"]),
                "key": f"user:{random.randint(1, 10000)}:profile",
                "result": random.choice(["HIT", "MISS", "OK", "EXPIRED"])
            }
        },
    ]
    
    # Calculate total weight
    total_weight = sum(t["weight"] for t in templates)
    
    for i in range(count):
        # Select template based on weights
        r = random.random() * total_weight
        cumulative = 0
        selected = templates[0]
        for t in templates:
            cumulative += t["weight"]
            if r <= cumulative:
                selected = t
                break
        
        # Generate log entry
        params = selected["params"]()
        message = selected["template"].format(**params)
        
        # Add timestamp with some progression
        timestamp = base_time + timedelta(minutes=i * 2, seconds=random.randint(0, 60))
        
        logs.append({
            "timestamp": timestamp.isoformat() + "Z",
            "level": selected["level"],
            "service": selected["service"],
            "message": message,
            "template_id": hashlib.md5(selected["template"].encode()).hexdigest()[:8]
        })
    
    return logs


# =============================================================================
# SECTION 1: TEMPLATE EXTRACTION (LOG PARSING)
# =============================================================================

class DrainParser:
    """
    Implementation of the Drain log parsing algorithm.
    
    Drain is a fixed-depth tree parser that efficiently extracts
    log templates by grouping similar log messages.
    
    Reference: He et al., "Drain: An Online Log Parsing Approach 
    with Fixed Depth Tree" (2017)
    """
    
    def __init__(
        self,
        depth: int = 4,
        similarity_threshold: float = 0.4,
        max_children: int = 100
    ):
        self.depth = depth
        self.similarity_threshold = similarity_threshold
        self.max_children = max_children
        self.root = {}
        self.templates = {}  # template_id -> TemplateInfo
        self.log_count = 0
    
    def _preprocess(self, message: str) -> List[str]:
        """Tokenize and preprocess log message."""
        # Replace common variable patterns with placeholders
        message = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', message)
        message = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', message)
        message = re.sub(r'[0-9a-fA-F]{8,}', '<HEX>', message)
        message = re.sub(r'\b\d+\b', '<NUM>', message)
        
        # Tokenize
        tokens = message.split()
        return tokens
    
    def _get_similarity(self, tokens: List[str], template_tokens: List[str]) -> float:
        """Calculate similarity between log tokens and template."""
        if len(tokens) != len(template_tokens):
            return 0.0
        
        matches = sum(1 for t1, t2 in zip(tokens, template_tokens) 
                     if t1 == t2 or t2 == '<*>')
        return matches / len(tokens)
    
    def _create_template(self, tokens: List[str]) -> str:
        """Create template string from tokens."""
        return ' '.join(tokens)
    
    def _update_template(self, tokens: List[str], template_tokens: List[str]) -> List[str]:
        """Update template by marking differing positions as wildcards."""
        new_template = []
        for t1, t2 in zip(tokens, template_tokens):
            if t1 == t2:
                new_template.append(t1)
            else:
                new_template.append('<*>')
        return new_template
    
    def parse(self, message: str) -> Tuple[str, str]:
        """
        Parse a log message and return its template.
        
        Returns:
            Tuple of (template_id, template_string)
        """
        tokens = self._preprocess(message)
        
        if not tokens:
            return "empty", ""
        
        # Navigate tree by length and first tokens
        length = len(tokens)
        
        if length not in self.root:
            self.root[length] = {}
        
        current = self.root[length]
        
        # Navigate by first few tokens (up to depth)
        for i in range(min(self.depth - 1, len(tokens))):
            token = tokens[i]
            if token not in current:
                if len(current) >= self.max_children:
                    token = '<*>'
                if token not in current:
                    current[token] = {}
            current = current[token]
        
        # Find matching template in leaf node
        if '_templates' not in current:
            current['_templates'] = []
        
        best_match = None
        best_similarity = 0.0
        
        for template_id, template_tokens in current['_templates']:
            similarity = self._get_similarity(tokens, template_tokens)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_match = (template_id, template_tokens)
                best_similarity = similarity
        
        if best_match:
            # Update existing template
            template_id, template_tokens = best_match
            new_template = self._update_template(tokens, template_tokens)
            
            # Update in place
            for i, (tid, _) in enumerate(current['_templates']):
                if tid == template_id:
                    current['_templates'][i] = (template_id, new_template)
                    break
            
            self.templates[template_id]['count'] += 1
            self.templates[template_id]['template'] = self._create_template(new_template)
            
        else:
            # Create new template
            template_id = f"T{len(self.templates):04d}"
            current['_templates'].append((template_id, tokens.copy()))
            self.templates[template_id] = {
                'template': self._create_template(tokens),
                'count': 1,
                'first_seen': self.log_count
            }
        
        self.log_count += 1
        return template_id, self.templates[template_id]['template']
    
    def get_templates(self) -> Dict[str, Dict]:
        """Get all discovered templates with statistics."""
        return self.templates.copy()
    
    def get_template_stats(self) -> List[Dict]:
        """Get templates sorted by frequency."""
        stats = []
        for tid, info in self.templates.items():
            stats.append({
                'template_id': tid,
                'template': info['template'],
                'count': info['count'],
                'percentage': info['count'] / self.log_count * 100 if self.log_count > 0 else 0
            })
        return sorted(stats, key=lambda x: x['count'], reverse=True)


def demo_template_extraction():
    """Demonstrate template extraction / log parsing."""
    print("\n" + "="*70)
    print("SECTION 1: TEMPLATE EXTRACTION (LOG PARSING)")
    print("="*70)
    
    # Generate sample logs
    print("\nGenerating sample logs...")
    logs = generate_sample_logs(300)
    print(f"Generated {len(logs)} log entries")
    
    # Parse logs with Drain
    parser = DrainParser(depth=4, similarity_threshold=0.4)
    
    print("\nExtracting templates with Drain algorithm...")
    for log in logs:
        parser.parse(log["message"])
    
    # Show results
    stats = parser.get_template_stats()
    print(f"\nDiscovered {len(stats)} unique templates:")
    print("-" * 70)
    
    for stat in stats[:10]:  # Top 10 templates
        print(f"\n📋 Template {stat['template_id']} ({stat['count']} logs, {stat['percentage']:.1f}%)")
        print(f"   {stat['template'][:70]}...")
    
    print(f"\n... and {len(stats) - 10} more templates")
    
    return parser, logs


# =============================================================================
# SECTION 2: SEMANTIC CLUSTERING
# =============================================================================

class BedrockEmbeddings:
    """Generate embeddings using Amazon Bedrock Titan model."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.client = get_bedrock_client(region_name)
        self.model_id = EMBEDDING_MODEL_ID
        self._cache = {}
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding with caching."""
        if text in self._cache:
            return self._cache[text]
        
        body = json.dumps({
            "inputText": text[:2000],  # Truncate for safety
            "dimensions": 1024,
            "normalize": True
        })
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response["body"].read())
        embedding = np.array(response_body["embedding"])
        self._cache[text] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed_text(t) for t in texts])


@dataclass
class SemanticCluster:
    """Represents a semantic cluster of logs."""
    cluster_id: int
    centroid: np.ndarray
    log_indices: List[int] = field(default_factory=list)
    label: str = ""
    representative_logs: List[str] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        return len(self.log_indices)


class SemanticLogClusterer:
    """
    Cluster logs by semantic similarity using embeddings.
    
    Unlike template extraction which groups by syntax,
    semantic clustering groups by meaning.
    """
    
    def __init__(self, embedder: BedrockEmbeddings):
        self.embedder = embedder
        self.embeddings = None
        self.logs = None
        self.clusters: Dict[int, SemanticCluster] = {}
    
    def fit(
        self,
        logs: List[Dict],
        method: str = "kmeans",
        n_clusters: int = None,
        **kwargs
    ) -> Dict[int, SemanticCluster]:
        """
        Cluster logs using specified method.
        
        Args:
            logs: List of log dictionaries
            method: "kmeans", "dbscan", "hierarchical", or "hdbscan"
            n_clusters: Number of clusters (for kmeans/hierarchical)
            
        Returns:
            Dictionary of cluster_id -> SemanticCluster
        """
        self.logs = logs
        
        # Generate embeddings
        print(f"   Generating embeddings for {len(logs)} logs...")
        messages = [f"{log['level']} {log['service']}: {log['message']}" for log in logs]
        self.embeddings = self.embedder.embed_batch(messages)
        
        # Cluster based on method
        print(f"   Clustering with {method}...")
        
        if method == "kmeans":
            n_clusters = n_clusters or min(20, len(logs) // 10)
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(self.embeddings)
            
        elif method == "dbscan":
            # DBSCAN needs careful parameter tuning for embeddings
            eps = kwargs.get('eps', 0.3)
            min_samples = kwargs.get('min_samples', 3)
            model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = model.fit_predict(self.embeddings)
            
        elif method == "hierarchical":
            n_clusters = n_clusters or min(20, len(logs) // 10)
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
            labels = model.fit_predict(self.embeddings)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Build cluster objects
        self.clusters = {}
        for idx, label in enumerate(labels):
            if label not in self.clusters:
                self.clusters[label] = SemanticCluster(
                    cluster_id=label,
                    centroid=np.zeros(self.embeddings.shape[1]),
                    log_indices=[],
                    representative_logs=[]
                )
            self.clusters[label].log_indices.append(idx)
        
        # Calculate centroids and find representative logs
        for cluster_id, cluster in self.clusters.items():
            if cluster.log_indices:
                cluster_embeddings = self.embeddings[cluster.log_indices]
                cluster.centroid = np.mean(cluster_embeddings, axis=0)
                
                # Find most central logs as representatives
                distances = cosine_similarity(
                    cluster.centroid.reshape(1, -1),
                    cluster_embeddings
                )[0]
                top_indices = np.argsort(distances)[-3:][::-1]
                cluster.representative_logs = [
                    messages[cluster.log_indices[i]] for i in top_indices
                ]
        
        return self.clusters
    
    def find_outliers(self, threshold: float = 0.5) -> List[int]:
        """
        Find logs that don't fit well in any cluster.
        
        These outliers are often the most interesting logs to investigate.
        """
        outliers = []
        
        for idx in range(len(self.logs)):
            # Find distance to nearest cluster centroid
            max_similarity = 0.0
            for cluster in self.clusters.values():
                similarity = cosine_similarity(
                    self.embeddings[idx].reshape(1, -1),
                    cluster.centroid.reshape(1, -1)
                )[0][0]
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity < threshold:
                outliers.append(idx)
        
        return outliers
    
    def assign_new_log(self, log: Dict) -> Tuple[int, float]:
        """Assign a new log to the nearest cluster."""
        message = f"{log['level']} {log['service']}: {log['message']}"
        embedding = self.embedder.embed_text(message)
        
        best_cluster = -1
        best_similarity = 0.0
        
        for cluster_id, cluster in self.clusters.items():
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                cluster.centroid.reshape(1, -1)
            )[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id
        
        return best_cluster, best_similarity


def demo_semantic_clustering(logs: List[Dict]):
    """Demonstrate semantic clustering."""
    print("\n" + "="*70)
    print("SECTION 2: SEMANTIC CLUSTERING")
    print("="*70)
    
    embedder = BedrockEmbeddings()
    clusterer = SemanticLogClusterer(embedder)
    
    # Use a subset for demo (embedding is expensive)
    sample_logs = logs[:100]
    
    # Cluster with K-means
    print("\n--- K-Means Clustering ---")
    clusters = clusterer.fit(sample_logs, method="kmeans", n_clusters=8)
    
    print(f"\nDiscovered {len(clusters)} clusters:")
    for cluster_id, cluster in sorted(clusters.items(), key=lambda x: x[1].size, reverse=True):
        if cluster_id == -1:
            print(f"\n🔴 Noise/Outliers: {cluster.size} logs")
        else:
            print(f"\n📁 Cluster {cluster_id}: {cluster.size} logs")
            print(f"   Representative logs:")
            for rep in cluster.representative_logs[:2]:
                print(f"   • {rep[:65]}...")
    
    # Find outliers
    print("\n--- Outlier Detection ---")
    outliers = clusterer.find_outliers(threshold=0.4)
    print(f"Found {len(outliers)} outlier logs (don't fit well in any cluster)")
    for idx in outliers[:3]:
        log = sample_logs[idx]
        print(f"   🔍 [{log['level']}] {log['message'][:60]}...")
    
    return clusterer


# =============================================================================
# SECTION 3: CLUSTER EVOLUTION TRACKING
# =============================================================================

@dataclass
class ClusterSnapshot:
    """Snapshot of cluster state at a point in time."""
    timestamp: datetime
    cluster_sizes: Dict[str, int]
    cluster_centroids: Dict[str, np.ndarray]
    total_logs: int


class ClusterEvolutionTracker:
    """
    Track how clusters change over time.
    
    Detects:
    - Emerging clusters (new patterns)
    - Growing clusters (increasing problems)
    - Shrinking clusters (resolved issues)
    - Splitting/merging clusters
    """
    
    def __init__(self):
        self.snapshots: List[ClusterSnapshot] = []
        self.alerts: List[Dict] = []
    
    def add_snapshot(
        self,
        timestamp: datetime,
        clusters: Dict[int, SemanticCluster],
        total_logs: int
    ):
        """Add a new cluster snapshot."""
        snapshot = ClusterSnapshot(
            timestamp=timestamp,
            cluster_sizes={str(k): v.size for k, v in clusters.items()},
            cluster_centroids={str(k): v.centroid for k, v in clusters.items()},
            total_logs=total_logs
        )
        self.snapshots.append(snapshot)
    
    def detect_changes(
        self,
        growth_threshold: float = 0.5,
        shrink_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect significant changes between last two snapshots.
        
        Returns list of detected changes with severity and description.
        """
        if len(self.snapshots) < 2:
            return []
        
        prev = self.snapshots[-2]
        curr = self.snapshots[-1]
        changes = []
        
        # Check for new clusters
        for cluster_id in curr.cluster_sizes:
            if cluster_id not in prev.cluster_sizes:
                changes.append({
                    'type': 'emerging',
                    'cluster_id': cluster_id,
                    'size': curr.cluster_sizes[cluster_id],
                    'severity': 'medium',
                    'description': f'New cluster emerged with {curr.cluster_sizes[cluster_id]} logs'
                })
        
        # Check for disappeared clusters
        for cluster_id in prev.cluster_sizes:
            if cluster_id not in curr.cluster_sizes:
                changes.append({
                    'type': 'disappeared',
                    'cluster_id': cluster_id,
                    'prev_size': prev.cluster_sizes[cluster_id],
                    'severity': 'low',
                    'description': f'Cluster disappeared (was {prev.cluster_sizes[cluster_id]} logs)'
                })
        
        # Check for growth/shrinkage
        for cluster_id in curr.cluster_sizes:
            if cluster_id in prev.cluster_sizes:
                prev_size = prev.cluster_sizes[cluster_id]
                curr_size = curr.cluster_sizes[cluster_id]
                
                if prev_size > 0:
                    change_ratio = (curr_size - prev_size) / prev_size
                    
                    if change_ratio > growth_threshold:
                        changes.append({
                            'type': 'growing',
                            'cluster_id': cluster_id,
                            'prev_size': prev_size,
                            'curr_size': curr_size,
                            'change_percent': change_ratio * 100,
                            'severity': 'high' if change_ratio > 1.0 else 'medium',
                            'description': f'Cluster grew by {change_ratio*100:.0f}%'
                        })
                    elif change_ratio < -shrink_threshold:
                        changes.append({
                            'type': 'shrinking',
                            'cluster_id': cluster_id,
                            'prev_size': prev_size,
                            'curr_size': curr_size,
                            'change_percent': change_ratio * 100,
                            'severity': 'low',
                            'description': f'Cluster shrank by {abs(change_ratio)*100:.0f}%'
                        })
        
        return changes
    
    def get_cluster_history(self, cluster_id: str) -> List[Tuple[datetime, int]]:
        """Get size history for a specific cluster."""
        history = []
        for snapshot in self.snapshots:
            size = snapshot.cluster_sizes.get(cluster_id, 0)
            history.append((snapshot.timestamp, size))
        return history


def demo_cluster_evolution():
    """Demonstrate cluster evolution tracking."""
    print("\n" + "="*70)
    print("SECTION 3: CLUSTER EVOLUTION TRACKING")
    print("="*70)
    
    tracker = ClusterEvolutionTracker()
    
    # Simulate cluster snapshots over time
    # Time 1: Initial state
    class MockCluster:
        def __init__(self, size):
            self.size = size
            self.centroid = np.random.rand(10)
    
    print("\n--- Simulating Cluster Evolution ---")
    
    # Snapshot 1
    clusters_t1 = {
        0: MockCluster(100),  # API errors
        1: MockCluster(50),   # Auth errors
        2: MockCluster(30),   # DB errors
    }
    tracker.add_snapshot(datetime(2024, 1, 15, 10, 0), clusters_t1, 180)
    print("T1: Initial state - 3 clusters (100, 50, 30 logs)")
    
    # Snapshot 2 - DB errors growing
    clusters_t2 = {
        0: MockCluster(105),  # Slight growth
        1: MockCluster(48),   # Stable
        2: MockCluster(75),   # Growing! (was 30)
    }
    tracker.add_snapshot(datetime(2024, 1, 15, 11, 0), clusters_t2, 228)
    print("T2: DB errors cluster growing (30 → 75)")
    
    # Detect changes
    changes = tracker.detect_changes(growth_threshold=0.3)
    print(f"\n📊 Detected {len(changes)} significant changes:")
    for change in changes:
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[change['severity']]
        print(f"   {severity_icon} [{change['type'].upper()}] Cluster {change['cluster_id']}: {change['description']}")
    
    # Snapshot 3 - New cluster emerges
    clusters_t3 = {
        0: MockCluster(110),
        1: MockCluster(45),
        2: MockCluster(90),   # Still growing
        3: MockCluster(25),   # NEW cluster!
    }
    tracker.add_snapshot(datetime(2024, 1, 15, 12, 0), clusters_t3, 270)
    print("\nT3: New cluster emerges (25 logs)")
    
    changes = tracker.detect_changes()
    print(f"\n📊 Detected {len(changes)} significant changes:")
    for change in changes:
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[change['severity']]
        print(f"   {severity_icon} [{change['type'].upper()}] Cluster {change['cluster_id']}: {change['description']}")
    
    # Show cluster history
    print("\n--- Cluster History ---")
    history = tracker.get_cluster_history("2")
    print("DB Errors cluster (ID: 2) over time:")
    for ts, size in history:
        print(f"   {ts.strftime('%H:%M')}: {size} logs {'📈' if size > 50 else ''}")
    
    return tracker


# =============================================================================
# SECTION 4: TREND DETECTION
# =============================================================================

class TrendDetector:
    """
    Detect trends in log patterns over time.
    
    Identifies:
    - Volume trends (increasing/decreasing error rates)
    - Seasonal patterns (periodic spikes)
    - Change points (sudden shifts)
    - Gradual drifts
    """
    
    @staticmethod
    def detect_linear_trend(
        timestamps: List[datetime],
        values: List[float]
    ) -> Dict:
        """
        Detect linear trend using regression.
        
        Returns trend direction, slope, and confidence.
        """
        if len(values) < 3:
            return {'trend': 'insufficient_data'}
        
        # Convert to numeric (minutes from start)
        start = min(timestamps)
        x = np.array([(t - start).total_seconds() / 60 for t in timestamps])
        y = np.array(values)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if p_value > 0.05:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'confidence': 1 - p_value,
            'change_per_hour': slope * 60
        }
    
    @staticmethod
    def detect_change_points(
        values: List[float],
        threshold: float = 2.0
    ) -> List[int]:
        """
        Detect sudden changes in the time series.
        
        Uses z-score to identify points where the value
        deviates significantly from the local mean.
        """
        if len(values) < 5:
            return []
        
        values = np.array(values)
        change_points = []
        
        # Sliding window approach
        window_size = min(5, len(values) // 3)
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                z_score = abs(values[i] - mean) / std
                if z_score > threshold:
                    change_points.append(i)
        
        return change_points
    
    @staticmethod
    def detect_periodicity(
        values: List[float],
        min_period: int = 2,
        max_period: int = None
    ) -> Dict:
        """
        Detect periodic patterns in the data.
        
        Useful for finding daily/weekly patterns or
        scheduled job impacts.
        """
        if len(values) < 10:
            return {'periodic': False}
        
        values = np.array(values)
        max_period = max_period or len(values) // 2
        
        # Autocorrelation
        autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation
        peaks, properties = find_peaks(autocorr[min_period:max_period], height=0.3)
        
        if len(peaks) > 0:
            # Adjust for offset
            peaks = peaks + min_period
            strongest_peak = peaks[np.argmax(properties['peak_heights'])]
            
            return {
                'periodic': True,
                'period': int(strongest_peak),
                'strength': float(autocorr[strongest_peak]),
                'all_periods': peaks.tolist()
            }
        
        return {'periodic': False}
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict:
        """Calculate summary statistics for a time series."""
        values = np.array(values)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'percentile_95': float(np.percentile(values, 95)),
            'percentile_99': float(np.percentile(values, 99))
        }


def demo_trend_detection(logs: List[Dict]):
    """Demonstrate trend detection."""
    print("\n" + "="*70)
    print("SECTION 4: TREND DETECTION")
    print("="*70)
    
    detector = TrendDetector()
    
    # Aggregate logs into time buckets
    print("\n--- Aggregating Logs by Time Window ---")
    
    # Parse timestamps and bucket by hour
    time_buckets = defaultdict(lambda: {'total': 0, 'errors': 0, 'warnings': 0})
    
    for log in logs:
        # Parse timestamp
        ts = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
        bucket = ts.replace(minute=0, second=0, microsecond=0)
        
        time_buckets[bucket]['total'] += 1
        if log['level'] == 'ERROR':
            time_buckets[bucket]['errors'] += 1
        elif log['level'] == 'WARN':
            time_buckets[bucket]['warnings'] += 1
    
    # Convert to sorted lists
    sorted_buckets = sorted(time_buckets.items())
    timestamps = [b[0] for b in sorted_buckets]
    totals = [b[1]['total'] for b in sorted_buckets]
    errors = [b[1]['errors'] for b in sorted_buckets]
    
    print(f"Analyzed {len(timestamps)} time buckets")
    
    # Detect linear trend
    print("\n--- Linear Trend Analysis ---")
    trend_result = detector.detect_linear_trend(timestamps, errors)
    print(f"Error trend: {trend_result['trend'].upper()}")
    if trend_result['trend'] != 'insufficient_data':
        print(f"   Slope: {trend_result['slope']:.4f} errors/minute")
        print(f"   Change per hour: {trend_result['change_per_hour']:.2f} errors")
        print(f"   Confidence: {trend_result['confidence']*100:.1f}%")
    
    # Detect change points
    print("\n--- Change Point Detection ---")
    # Simulate a time series with a change point
    simulated_values = [10, 12, 11, 13, 10, 11, 45, 48, 52, 47, 50, 49]
    change_points = detector.detect_change_points(simulated_values, threshold=2.0)
    print(f"Simulated data: {simulated_values}")
    print(f"Change points detected at indices: {change_points}")
    
    # Detect periodicity
    print("\n--- Periodicity Detection ---")
    # Simulate periodic data (period of 4)
    periodic_values = [10, 20, 15, 25] * 5  # Repeating pattern
    period_result = detector.detect_periodicity(periodic_values)
    print(f"Simulated data: {periodic_values[:8]}... (repeating)")
    print(f"Periodic: {period_result['periodic']}")
    if period_result['periodic']:
        print(f"   Period: {period_result['period']} time units")
        print(f"   Strength: {period_result['strength']:.2f}")
    
    # Statistics
    print("\n--- Summary Statistics ---")
    stats = detector.calculate_statistics(errors)
    print(f"Error statistics:")
    print(f"   Mean: {stats['mean']:.1f} errors/bucket")
    print(f"   Std Dev: {stats['std']:.1f}")
    print(f"   Range: {stats['min']:.0f} - {stats['max']:.0f}")
    print(f"   95th percentile: {stats['percentile_95']:.1f}")
    
    return detector


# =============================================================================
# SECTION 5: CORRELATION DISCOVERY
# =============================================================================

class CorrelationAnalyzer:
    """
    Discover correlations between log patterns.
    
    Finds:
    - Co-occurring error types
    - Cascading failures (A causes B)
    - Common root causes
    """
    
    def __init__(self):
        self.co_occurrences: Dict[Tuple[str, str], int] = defaultdict(int)
        self.sequences: List[List[str]] = []
    
    def analyze_co_occurrence(
        self,
        logs: List[Dict],
        window_seconds: int = 60
    ) -> Dict[Tuple[str, str], float]:
        """
        Find log types that frequently occur together.
        
        Uses a sliding time window to find co-occurring events.
        """
        # Group logs by template/type
        type_occurrences = defaultdict(list)
        
        for log in logs:
            ts = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
            log_type = f"{log['level']}:{log['service']}"
            type_occurrences[log_type].append(ts)
        
        # Find co-occurrences within time window
        types = list(type_occurrences.keys())
        co_occurrence_counts = defaultdict(int)
        individual_counts = defaultdict(int)
        
        for i, type1 in enumerate(types):
            individual_counts[type1] = len(type_occurrences[type1])
            for type2 in types[i+1:]:
                # Count how many times type1 and type2 appear in same window
                for ts1 in type_occurrences[type1]:
                    for ts2 in type_occurrences[type2]:
                        if abs((ts1 - ts2).total_seconds()) <= window_seconds:
                            co_occurrence_counts[(type1, type2)] += 1
        
        # Calculate correlation strength (Jaccard similarity)
        correlations = {}
        for (type1, type2), count in co_occurrence_counts.items():
            union = individual_counts[type1] + individual_counts[type2] - count
            if union > 0:
                correlations[(type1, type2)] = count / union
        
        return dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
    
    def find_cascades(
        self,
        logs: List[Dict],
        max_delay_seconds: int = 30
    ) -> List[Dict]:
        """
        Find potential cascade patterns (A → B).
        
        Looks for error types that consistently follow other events.
        """
        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x['timestamp'])
        
        # Track sequences
        sequence_counts = defaultdict(int)
        
        for i in range(len(sorted_logs) - 1):
            log1 = sorted_logs[i]
            ts1 = datetime.fromisoformat(log1['timestamp'].replace('Z', '+00:00'))
            type1 = f"{log1['level']}:{log1['service']}"
            
            # Look for following events
            for j in range(i + 1, min(i + 10, len(sorted_logs))):
                log2 = sorted_logs[j]
                ts2 = datetime.fromisoformat(log2['timestamp'].replace('Z', '+00:00'))
                
                delay = (ts2 - ts1).total_seconds()
                if delay > max_delay_seconds:
                    break
                
                type2 = f"{log2['level']}:{log2['service']}"
                if type1 != type2:
                    sequence_counts[(type1, type2)] += 1
        
        # Filter to significant sequences
        cascades = []
        for (type1, type2), count in sequence_counts.items():
            if count >= 3:  # Minimum occurrence threshold
                cascades.append({
                    'trigger': type1,
                    'effect': type2,
                    'count': count,
                    'description': f'{type1} often followed by {type2}'
                })
        
        return sorted(cascades, key=lambda x: x['count'], reverse=True)
    
    def find_common_context(
        self,
        logs: List[Dict],
        target_pattern: str
    ) -> Dict[str, int]:
        """
        Find what other patterns commonly appear with a target pattern.
        
        Useful for understanding the context around specific errors.
        """
        # Find indices of target pattern
        target_indices = []
        for i, log in enumerate(logs):
            if target_pattern.lower() in log['message'].lower():
                target_indices.append(i)
        
        # Collect context logs (surrounding each target)
        context_counts = defaultdict(int)
        window = 5  # Logs before and after
        
        for idx in target_indices:
            start = max(0, idx - window)
            end = min(len(logs), idx + window + 1)
            
            for i in range(start, end):
                if i != idx:
                    context_type = f"{logs[i]['level']}:{logs[i]['service']}"
                    context_counts[context_type] += 1
        
        return dict(sorted(context_counts.items(), key=lambda x: x[1], reverse=True))


def demo_correlation_analysis(logs: List[Dict]):
    """Demonstrate correlation analysis."""
    print("\n" + "="*70)
    print("SECTION 5: CORRELATION DISCOVERY")
    print("="*70)
    
    analyzer = CorrelationAnalyzer()
    
    # Co-occurrence analysis
    print("\n--- Co-occurrence Analysis ---")
    co_occurrences = analyzer.analyze_co_occurrence(logs, window_seconds=60)
    
    print("Top co-occurring log patterns (within 60s window):")
    for (type1, type2), strength in list(co_occurrences.items())[:5]:
        print(f"   🔗 {type1} ↔ {type2}")
        print(f"      Correlation strength: {strength:.3f}")
    
    # Cascade detection
    print("\n--- Cascade Pattern Detection ---")
    cascades = analyzer.find_cascades(logs, max_delay_seconds=30)
    
    print("Potential cascade patterns (A often followed by B):")
    for cascade in cascades[:5]:
        print(f"   ⚡ {cascade['trigger']} → {cascade['effect']}")
        print(f"      Occurred {cascade['count']} times")
    
    # Context analysis
    print("\n--- Context Analysis ---")
    # Find what happens around connection errors
    context = analyzer.find_common_context(logs, "Connection")
    print("Patterns commonly appearing around 'Connection' issues:")
    for pattern, count in list(context.items())[:5]:
        print(f"   📌 {pattern}: {count} occurrences")
    
    return analyzer


# =============================================================================
# SECTION 6: DIMENSIONALITY REDUCTION & VISUALIZATION
# =============================================================================

class LogVisualizer:
    """
    Reduce embedding dimensions for visualization.
    
    Enables plotting logs in 2D/3D to see cluster structure
    and identify outliers visually.
    """
    
    def __init__(self):
        self.pca = None
        self.reduced_embeddings = None
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        method: str = "pca"
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, n_features)
            n_components: Target dimensions (2 or 3)
            method: "pca" (fast, linear) or future support for UMAP
            
        Returns:
            Reduced embeddings (n_samples, n_components)
        """
        if method == "pca":
            self.pca = PCA(n_components=n_components)
            self.reduced_embeddings = self.pca.fit_transform(embeddings)
            
            print(f"   Explained variance: {sum(self.pca.explained_variance_ratio_)*100:.1f}%")
            
        return self.reduced_embeddings
    
    def get_cluster_boundaries(
        self,
        reduced_embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """Calculate cluster boundaries for visualization."""
        boundaries = {}
        
        for label in np.unique(labels):
            if label == -1:  # Skip noise
                continue
                
            mask = labels == label
            cluster_points = reduced_embeddings[mask]
            
            boundaries[label] = {
                'center': np.mean(cluster_points, axis=0).tolist(),
                'min': np.min(cluster_points, axis=0).tolist(),
                'max': np.max(cluster_points, axis=0).tolist(),
                'std': np.std(cluster_points, axis=0).tolist()
            }
        
        return boundaries
    
    def generate_ascii_plot(
        self,
        reduced_embeddings: np.ndarray,
        labels: np.ndarray,
        width: int = 60,
        height: int = 20
    ) -> str:
        """
        Generate a simple ASCII scatter plot.
        
        For actual visualization, use matplotlib or plotly.
        """
        if reduced_embeddings.shape[1] < 2:
            return "Need at least 2 dimensions for plotting"
        
        x = reduced_embeddings[:, 0]
        y = reduced_embeddings[:, 1]
        
        # Normalize to grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        x_norm = ((x - x_min) / (x_max - x_min) * (width - 1)).astype(int)
        y_norm = ((y - y_min) / (y_max - y_min) * (height - 1)).astype(int)
        
        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        markers = '0123456789ABCDEFGHIJ'
        for i in range(len(x_norm)):
            xi, yi = x_norm[i], height - 1 - y_norm[i]  # Flip y
            label = labels[i] if labels[i] >= 0 else -1
            marker = markers[label % len(markers)] if label >= 0 else '.'
            grid[yi][xi] = marker
        
        # Convert to string
        lines = ['┌' + '─' * width + '┐']
        for row in grid:
            lines.append('│' + ''.join(row) + '│')
        lines.append('└' + '─' * width + '┘')
        
        return '\n'.join(lines)


def demo_visualization(clusterer: SemanticLogClusterer):
    """Demonstrate dimensionality reduction and visualization."""
    print("\n" + "="*70)
    print("SECTION 6: DIMENSIONALITY REDUCTION & VISUALIZATION")
    print("="*70)
    
    if clusterer.embeddings is None:
        print("No embeddings available for visualization")
        return
    
    visualizer = LogVisualizer()
    
    # Reduce dimensions
    print("\n--- PCA Dimensionality Reduction ---")
    print(f"Original dimensions: {clusterer.embeddings.shape[1]}")
    reduced = visualizer.reduce_dimensions(clusterer.embeddings, n_components=2)
    print(f"Reduced dimensions: {reduced.shape[1]}")
    
    # Get cluster labels
    labels = np.array([
        next((cid for cid, c in clusterer.clusters.items() if i in c.log_indices), -1)
        for i in range(len(clusterer.logs))
    ])
    
    # Generate ASCII plot
    print("\n--- Cluster Visualization (ASCII) ---")
    plot = visualizer.generate_ascii_plot(reduced, labels)
    print(plot)
    print("\nLegend: Numbers/letters = cluster IDs, dots = noise/outliers")
    
    # Cluster boundaries
    print("\n--- Cluster Boundaries ---")
    boundaries = visualizer.get_cluster_boundaries(reduced, labels)
    for label, bounds in list(boundaries.items())[:5]:
        print(f"Cluster {label}:")
        print(f"   Center: ({bounds['center'][0]:.2f}, {bounds['center'][1]:.2f})")
        print(f"   Spread (std): ({bounds['std'][0]:.2f}, {bounds['std'][1]:.2f})")
    
    return visualizer


# =============================================================================
# SECTION 7: PATTERN LIBRARY
# =============================================================================

@dataclass
class PatternDefinition:
    """A known pattern in the pattern library."""
    pattern_id: str
    name: str
    description: str
    template: str = None
    centroid: np.ndarray = None
    examples: List[str] = field(default_factory=list)
    occurrence_count: int = 0
    first_seen: datetime = None
    last_seen: datetime = None
    severity: str = "medium"
    recommended_action: str = ""
    tags: List[str] = field(default_factory=list)


class PatternLibrary:
    """
    A living library of known log patterns.
    
    Features:
    - Store and retrieve known patterns
    - Match new logs against known patterns
    - Track pattern statistics over time
    - Export/import for persistence
    """
    
    def __init__(self, embedder: BedrockEmbeddings = None):
        self.embedder = embedder
        self.patterns: Dict[str, PatternDefinition] = {}
    
    def add_pattern(
        self,
        name: str,
        description: str,
        examples: List[str],
        severity: str = "medium",
        recommended_action: str = "",
        tags: List[str] = None
    ) -> PatternDefinition:
        """Add a new pattern to the library."""
        pattern_id = f"PAT_{len(self.patterns):04d}"
        
        # Generate centroid from examples if embedder available
        centroid = None
        if self.embedder and examples:
            embeddings = self.embedder.embed_batch(examples)
            centroid = np.mean(embeddings, axis=0)
        
        pattern = PatternDefinition(
            pattern_id=pattern_id,
            name=name,
            description=description,
            examples=examples,
            centroid=centroid,
            severity=severity,
            recommended_action=recommended_action,
            tags=tags or [],
            first_seen=datetime.now(),
            last_seen=datetime.now()
        )
        
        self.patterns[pattern_id] = pattern
        return pattern
    
    def match_log(
        self,
        log_message: str,
        threshold: float = 0.6
    ) -> Optional[Tuple[PatternDefinition, float]]:
        """
        Match a log message against known patterns.
        
        Returns the best matching pattern and confidence score.
        """
        if not self.embedder:
            return None
        
        log_embedding = self.embedder.embed_text(log_message)
        
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns.values():
            if pattern.centroid is not None:
                similarity = cosine_similarity(
                    log_embedding.reshape(1, -1),
                    pattern.centroid.reshape(1, -1)
                )[0][0]
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = pattern
        
        if best_match:
            # Update statistics
            best_match.occurrence_count += 1
            best_match.last_seen = datetime.now()
            return best_match, best_score
        
        return None
    
    def get_pattern_stats(self) -> List[Dict]:
        """Get statistics for all patterns."""
        stats = []
        for pattern in self.patterns.values():
            stats.append({
                'pattern_id': pattern.pattern_id,
                'name': pattern.name,
                'occurrences': pattern.occurrence_count,
                'severity': pattern.severity,
                'last_seen': pattern.last_seen.isoformat() if pattern.last_seen else None,
                'tags': pattern.tags
            })
        return sorted(stats, key=lambda x: x['occurrences'], reverse=True)
    
    def export_library(self) -> Dict:
        """Export library to JSON-serializable format."""
        export = {}
        for pid, pattern in self.patterns.items():
            export[pid] = {
                'name': pattern.name,
                'description': pattern.description,
                'examples': pattern.examples,
                'severity': pattern.severity,
                'recommended_action': pattern.recommended_action,
                'tags': pattern.tags,
                'occurrence_count': pattern.occurrence_count
            }
        return export
    
    def import_library(self, data: Dict):
        """Import patterns from exported data."""
        for pid, pdata in data.items():
            self.add_pattern(
                name=pdata['name'],
                description=pdata['description'],
                examples=pdata['examples'],
                severity=pdata.get('severity', 'medium'),
                recommended_action=pdata.get('recommended_action', ''),
                tags=pdata.get('tags', [])
            )


def demo_pattern_library():
    """Demonstrate the pattern library."""
    print("\n" + "="*70)
    print("SECTION 7: PATTERN LIBRARY")
    print("="*70)
    
    embedder = BedrockEmbeddings()
    library = PatternLibrary(embedder)
    
    # Add known patterns
    print("\n--- Building Pattern Library ---")
    
    library.add_pattern(
        name="Database Connection Failure",
        description="Database connection issues including timeouts and refused connections",
        examples=[
            "Connection to db-primary:5432 failed: connection refused",
            "Database connection timeout after 5000ms",
            "Unable to acquire connection from pool"
        ],
        severity="high",
        recommended_action="Check database health, connection pool settings, and network connectivity",
        tags=["database", "connectivity", "critical"]
    )
    
    library.add_pattern(
        name="Authentication Failure",
        description="User authentication and authorization failures",
        examples=[
            "Authentication failed for user: invalid password",
            "Login rejected: account locked",
            "Token expired for session"
        ],
        severity="medium",
        recommended_action="Check for brute force attempts, review user account status",
        tags=["security", "authentication"]
    )
    
    library.add_pattern(
        name="Memory Pressure",
        description="High memory usage and potential OOM situations",
        examples=[
            "Memory usage high: 7500MB / 8192MB (92%)",
            "GC overhead limit exceeded",
            "OutOfMemoryError: Java heap space"
        ],
        severity="critical",
        recommended_action="Investigate memory leaks, consider scaling or heap adjustment",
        tags=["performance", "memory", "critical"]
    )
    
    print(f"Added {len(library.patterns)} patterns to library")
    
    # Match new logs
    print("\n--- Matching New Logs Against Library ---")
    test_logs = [
        "Connection to postgres-replica:5432 failed: timeout after 30s",
        "User alice@example.com authentication failed: wrong password",
        "Request completed successfully in 45ms",
        "Memory usage at 95%, triggering GC",
    ]
    
    for log in test_logs:
        print(f"\n📝 \"{log[:60]}...\"")
        match = library.match_log(log)
        if match:
            pattern, score = match
            print(f"   ✅ Matched: {pattern.name} (confidence: {score:.2f})")
            print(f"   📋 Severity: {pattern.severity}")
            print(f"   💡 Action: {pattern.recommended_action[:50]}...")
        else:
            print(f"   ❓ No matching pattern (novel event?)")
    
    # Show library stats
    print("\n--- Pattern Library Statistics ---")
    stats = library.get_pattern_stats()
    for stat in stats:
        print(f"   📊 {stat['name']}: {stat['occurrences']} matches, severity={stat['severity']}")
    
    return library


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run all demonstrations.
    
    Sections:
    1. Template Extraction - Parse logs into templates
    2. Semantic Clustering - Group by meaning
    3. Cluster Evolution - Track changes over time
    4. Trend Detection - Find patterns in time series
    5. Correlation Discovery - Find related events
    6. Visualization - Reduce dimensions for plotting
    7. Pattern Library - Build institutional knowledge
    """
    print("="*70)
    print("CHAPTER 5: PATTERN RECOGNITION AT SCALE")
    print("Code Examples with Amazon Bedrock")
    print("="*70)
    
    try:
        # Section 1: Template Extraction
        parser, logs = demo_template_extraction()
        
        # Section 2: Semantic Clustering
        clusterer = demo_semantic_clustering(logs)
        
        # Section 3: Cluster Evolution
        demo_cluster_evolution()
        
        # Section 4: Trend Detection
        demo_trend_detection(logs)
        
        # Section 5: Correlation Analysis
        demo_correlation_analysis(logs)
        
        # Section 6: Visualization
        demo_visualization(clusterer)
        
        # Section 7: Pattern Library
        demo_pattern_library()
        
        print("\n" + "="*70)
        print("✅ All demonstrations completed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure AWS credentials are configured")
        print("2. Enable Bedrock models in AWS Console")
        print("3. Check region supports Bedrock")
        raise


if __name__ == "__main__":
    main()
