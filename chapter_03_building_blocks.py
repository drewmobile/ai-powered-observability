"""
Chapter 3: Building Blocks — The AI Toolkit for Ops
=====================================================

Code examples demonstrating the core AI/ML building blocks for observability:
- Embeddings with Amazon Bedrock Titan
- Vector search and similarity
- Large Language Models with Claude via Bedrock
- Anomaly detection algorithms
- Classification and clustering

Prerequisites:
    pip install boto3 numpy scikit-learn

AWS Configuration:
    Ensure you have AWS credentials configured with access to Amazon Bedrock.
    Enable the following models in your Bedrock console:
    - amazon.titan-embed-text-v2:0 (for embeddings)
    - anthropic.claude-3-sonnet-20240229-v1:0 (for LLM tasks)
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import boto3
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# CONFIGURATION
# =============================================================================

# Amazon Bedrock model IDs
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# Initialize Bedrock client
def get_bedrock_client(region_name: str = "us-east-1"):
    """Create a Bedrock runtime client."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name
    )


# =============================================================================
# SAMPLE OBSERVABILITY DATA
# =============================================================================

SAMPLE_LOGS = [
    # Authentication failures
    "2024-01-15 10:23:45 ERROR auth-service: Login failed for user john@example.com - invalid password",
    "2024-01-15 10:24:01 ERROR auth-service: Authentication rejected - token expired for session abc123",
    "2024-01-15 10:24:15 WARN auth-service: Too many failed login attempts from IP 192.168.1.100",
    "2024-01-15 10:25:00 ERROR auth-service: Permission denied - user lacks required role: admin",
    
    # Database issues
    "2024-01-15 10:26:30 ERROR db-service: Connection refused to postgres://db-primary:5432",
    "2024-01-15 10:26:45 ERROR db-service: Query timeout after 30000ms - SELECT * FROM orders",
    "2024-01-15 10:27:00 WARN db-service: Connection pool exhausted, waiting for available connection",
    "2024-01-15 10:27:15 ERROR db-service: Deadlock detected in transaction txn_789",
    
    # Memory issues
    "2024-01-15 10:28:00 CRITICAL app-server: OutOfMemoryError - Java heap space exceeded",
    "2024-01-15 10:28:05 WARN app-server: GC overhead limit exceeded - 98% time spent in GC",
    "2024-01-15 10:28:10 ERROR app-server: Memory allocation failed for 512MB buffer",
    
    # Network issues
    "2024-01-15 10:29:00 ERROR api-gateway: Connection timeout to payment-service:8080",
    "2024-01-15 10:29:15 ERROR api-gateway: Socket timeout - no response from inventory-service",
    "2024-01-15 10:29:30 WARN api-gateway: High latency detected: 2500ms to shipping-service",
    
    # Success messages (for contrast)
    "2024-01-15 10:30:00 INFO auth-service: User alice@example.com logged in successfully",
    "2024-01-15 10:30:15 INFO api-gateway: Request completed successfully in 45ms",
    "2024-01-15 10:30:30 INFO db-service: Database backup completed - 2.3GB archived",
]

# Sample metrics time series (CPU utilization)
SAMPLE_METRICS = {
    "timestamps": [
        "2024-01-15T10:00:00", "2024-01-15T10:05:00", "2024-01-15T10:10:00",
        "2024-01-15T10:15:00", "2024-01-15T10:20:00", "2024-01-15T10:25:00",
        "2024-01-15T10:30:00", "2024-01-15T10:35:00", "2024-01-15T10:40:00",
        "2024-01-15T10:45:00", "2024-01-15T10:50:00", "2024-01-15T10:55:00",
    ],
    "cpu_percent": [45, 48, 52, 47, 95, 92, 88, 51, 49, 46, 98, 45],  # Anomalies at indices 4,5,6,10
    "memory_percent": [62, 64, 63, 65, 78, 82, 85, 66, 64, 63, 89, 64],
    "request_latency_ms": [120, 125, 118, 122, 450, 520, 380, 130, 128, 125, 620, 122],
}


# =============================================================================
# SECTION 1: EMBEDDINGS WITH AMAZON BEDROCK
# =============================================================================

class BedrockEmbeddings:
    """
    Generate embeddings using Amazon Bedrock Titan Embeddings model.
    
    Embeddings convert text into numerical vectors that capture semantic meaning.
    Similar texts produce similar vectors, enabling semantic search and clustering.
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        self.client = get_bedrock_client(region_name)
        self.model_id = EMBEDDING_MODEL_ID
        self.embedding_dimension = 1024  # Titan v2 produces 1024-dimensional vectors
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            A numpy array of shape (1024,) representing the embedding
        """
        body = json.dumps({
            "inputText": text,
            "dimensions": self.embedding_dimension,
            "normalize": True  # Normalize for cosine similarity
        })
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response["body"].read())
        embedding = np.array(response_body["embedding"])
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Note: Titan doesn't support native batching, so we process sequentially.
        For production, consider parallelization or using a batching service.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A numpy array of shape (n_texts, 1024)
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return np.array(embeddings)


def demo_embeddings():
    """Demonstrate embedding generation and semantic similarity."""
    print("\n" + "="*70)
    print("SECTION 1: EMBEDDINGS DEMONSTRATION")
    print("="*70)
    
    embedder = BedrockEmbeddings()
    
    # Embed sample logs
    print("\nGenerating embeddings for sample log messages...")
    test_logs = [
        "Connection refused to database server",
        "Database connection timeout after 30 seconds",
        "User authentication failed - invalid password",
        "Login successful for user admin",
    ]
    
    embeddings = embedder.embed_batch(test_logs)
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Calculate similarity matrix
    print("\nSemantic Similarity Matrix:")
    print("-" * 50)
    similarity_matrix = cosine_similarity(embeddings)
    
    for i, log1 in enumerate(test_logs):
        print(f"\nLog {i+1}: '{log1[:50]}...'")
        for j, log2 in enumerate(test_logs):
            if i < j:
                sim = similarity_matrix[i][j]
                print(f"  → Similarity to Log {j+1}: {sim:.3f}")
    
    print("\n💡 Key insight: Database-related logs (1,2) have higher similarity")
    print("   than database logs compared to authentication logs (3,4).")
    
    return embedder, embeddings


# =============================================================================
# SECTION 2: VECTOR DATABASE (IN-MEMORY IMPLEMENTATION)
# =============================================================================

@dataclass
class VectorRecord:
    """A record in our vector store."""
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict


class SimpleVectorStore:
    """
    A simple in-memory vector store for demonstration.
    
    In production, use a dedicated vector database like:
    - Amazon OpenSearch with vector search
    - Pinecone
    - Weaviate
    - Milvus
    - pgvector (PostgreSQL extension)
    
    This implementation uses brute-force search. Production systems use
    approximate nearest neighbor (ANN) algorithms like HNSW for speed.
    """
    
    def __init__(self, embedder: BedrockEmbeddings):
        self.embedder = embedder
        self.records: List[VectorRecord] = []
    
    def add(self, id: str, text: str, metadata: Dict = None):
        """Add a document to the vector store."""
        embedding = self.embedder.embed_text(text)
        record = VectorRecord(
            id=id,
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.records.append(record)
    
    def add_batch(self, documents: List[Dict]):
        """
        Add multiple documents to the vector store.
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
        """
        for doc in documents:
            self.add(doc["id"], doc["text"], doc.get("metadata", {}))
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of (record, similarity_score) tuples, sorted by similarity
        """
        query_embedding = self.embedder.embed_text(query)
        
        # Calculate similarities
        results = []
        for record in self.records:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                record.embedding.reshape(1, -1)
            )[0][0]
            results.append((record, similarity))
        
        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def find_similar(self, record_id: str, top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        """Find records similar to an existing record."""
        target_record = next((r for r in self.records if r.id == record_id), None)
        if not target_record:
            raise ValueError(f"Record {record_id} not found")
        
        results = []
        for record in self.records:
            if record.id == record_id:
                continue
            similarity = cosine_similarity(
                target_record.embedding.reshape(1, -1),
                record.embedding.reshape(1, -1)
            )[0][0]
            results.append((record, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def demo_vector_search(embedder: BedrockEmbeddings):
    """Demonstrate vector search capabilities."""
    print("\n" + "="*70)
    print("SECTION 2: VECTOR SEARCH DEMONSTRATION")
    print("="*70)
    
    # Create vector store and add sample logs
    store = SimpleVectorStore(embedder)
    
    print("\nIndexing sample logs into vector store...")
    for i, log in enumerate(SAMPLE_LOGS):
        # Extract log level from the log message
        level = "INFO"
        for lvl in ["CRITICAL", "ERROR", "WARN", "INFO", "DEBUG"]:
            if lvl in log:
                level = lvl
                break
        
        store.add(
            id=f"log_{i}",
            text=log,
            metadata={"index": i, "level": level}
        )
    
    print(f"Indexed {len(store.records)} log messages")
    
    # Semantic search examples
    print("\n--- Semantic Search Demo ---")
    queries = [
        "authentication problems",
        "database connectivity issues",
        "system running out of memory",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = store.search(query, top_k=3)
        for record, score in results:
            print(f"   [{score:.3f}] {record.text[:70]}...")
    
    # Find similar logs
    print("\n--- Similar Log Discovery ---")
    print("\nFinding logs similar to the first authentication error...")
    similar = store.find_similar("log_0", top_k=3)
    print(f"Base log: {store.records[0].text[:60]}...")
    for record, score in similar:
        print(f"   [{score:.3f}] {record.text[:60]}...")
    
    return store


# =============================================================================
# SECTION 3: LARGE LANGUAGE MODELS WITH CLAUDE VIA BEDROCK
# =============================================================================

class BedrockLLM:
    """
    Interact with Claude via Amazon Bedrock for observability tasks.
    
    LLMs enable:
    - Log summarization
    - Error explanation
    - Root cause analysis suggestions
    - Natural language querying
    - Alert enrichment
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        self.client = get_bedrock_client(region_name)
        self.model_id = LLM_MODEL_ID
    
    def invoke(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.3
    ) -> str:
        """
        Send a prompt to Claude and get a response.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            max_tokens: Maximum response length
            temperature: Randomness (0=deterministic, 1=creative)
            
        Returns:
            The model's response text
        """
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
    
    def summarize_logs(self, logs: List[str]) -> str:
        """
        Summarize a collection of log entries.
        
        This is useful for:
        - Incident summaries
        - Daily/weekly reports
        - On-call handoffs
        """
        system_prompt = """You are an expert SRE analyzing system logs. 
        Provide concise, actionable summaries focusing on:
        1. Key issues identified
        2. Affected systems/services
        3. Potential impact
        4. Recommended actions"""
        
        logs_text = "\n".join(logs)
        prompt = f"""Analyze and summarize the following log entries:

{logs_text}

Provide a brief summary highlighting the main issues and their severity."""
        
        return self.invoke(prompt, system_prompt)
    
    def explain_error(self, error_log: str, context_logs: List[str] = None) -> str:
        """
        Explain an error message in plain language.
        
        Useful for:
        - Helping junior engineers understand errors
        - Documentation
        - Incident reports
        """
        system_prompt = """You are a helpful SRE explaining technical errors.
        Provide clear explanations that:
        1. Explain what the error means
        2. List common causes
        3. Suggest debugging steps
        Keep explanations concise and practical."""
        
        context = ""
        if context_logs:
            context = f"\n\nRelated log entries:\n" + "\n".join(context_logs)
        
        prompt = f"""Explain this error in plain language:

{error_log}
{context}

What does this error mean and what might have caused it?"""
        
        return self.invoke(prompt, system_prompt)
    
    def suggest_root_cause(self, logs: List[str], metrics: Dict = None) -> str:
        """
        Analyze logs and metrics to suggest potential root causes.
        
        This demonstrates correlation reasoning - combining multiple
        signals to identify underlying issues.
        """
        system_prompt = """You are an expert SRE performing root cause analysis.
        Analyze the provided data and:
        1. Identify patterns and correlations
        2. Suggest the most likely root cause
        3. Explain your reasoning
        4. Recommend next steps for investigation"""
        
        prompt = f"Analyze these log entries:\n\n"
        prompt += "\n".join(logs)
        
        if metrics:
            prompt += f"\n\nRelevant metrics:\n{json.dumps(metrics, indent=2)}"
        
        prompt += "\n\nWhat is the most likely root cause of these issues?"
        
        return self.invoke(prompt, system_prompt)
    
    def translate_to_query(self, natural_language: str, query_language: str = "SQL") -> str:
        """
        Translate natural language to a technical query.
        
        Enables non-technical users to query observability data
        without knowing specific query languages.
        """
        system_prompt = f"""You are an expert at translating natural language 
        to {query_language} queries for observability systems.
        Return only the query, no explanation."""
        
        prompt = f"""Translate this request to a {query_language} query:

"{natural_language}"

Assume a logs table with columns: timestamp, level, service, message"""
        
        return self.invoke(prompt, system_prompt, temperature=0.1)


def demo_llm_capabilities():
    """Demonstrate LLM capabilities for observability."""
    print("\n" + "="*70)
    print("SECTION 3: LARGE LANGUAGE MODEL DEMONSTRATION")
    print("="*70)
    
    llm = BedrockLLM()
    
    # Log summarization
    print("\n--- Log Summarization ---")
    error_logs = [log for log in SAMPLE_LOGS if "ERROR" in log or "CRITICAL" in log]
    print(f"\nSummarizing {len(error_logs)} error logs...")
    summary = llm.summarize_logs(error_logs)
    print(f"\n📋 Summary:\n{summary}")
    
    # Error explanation
    print("\n--- Error Explanation ---")
    error_log = "2024-01-15 10:28:00 CRITICAL app-server: OutOfMemoryError - Java heap space exceeded"
    print(f"\nExplaining: {error_log}")
    explanation = llm.explain_error(error_log)
    print(f"\n📖 Explanation:\n{explanation}")
    
    # Root cause analysis
    print("\n--- Root Cause Analysis ---")
    incident_logs = SAMPLE_LOGS[8:11]  # Memory-related logs
    metrics_snapshot = {
        "cpu_percent": 92,
        "memory_percent": 98,
        "heap_used_mb": 7800,
        "heap_max_mb": 8192
    }
    print("\nAnalyzing memory-related incident...")
    root_cause = llm.suggest_root_cause(incident_logs, metrics_snapshot)
    print(f"\n🔍 Root Cause Analysis:\n{root_cause}")
    
    # Natural language to query
    print("\n--- Natural Language to Query ---")
    nl_queries = [
        "Show me all authentication errors from the last hour",
        "Find database timeout errors grouped by service",
    ]
    for query in nl_queries:
        print(f"\n💬 '{query}'")
        sql = llm.translate_to_query(query, "SQL")
        print(f"📝 SQL: {sql}")
    
    return llm


# =============================================================================
# SECTION 4: ANOMALY DETECTION
# =============================================================================

class AnomalyDetector:
    """
    Anomaly detection methods for observability data.
    
    Covers both statistical and ML-based approaches:
    - Z-score (statistical)
    - IQR method (statistical)
    - Isolation Forest (ML)
    - Embedding-based anomaly detection
    """
    
    @staticmethod
    def zscore_detection(
        data: np.ndarray,
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score method.
        
        Simple and interpretable. Works well for normally distributed data.
        A data point is anomalous if its Z-score exceeds the threshold.
        
        Args:
            data: 1D array of values
            threshold: Number of standard deviations (default 3.0)
            
        Returns:
            Tuple of (anomaly_indices, z_scores)
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return np.array([]), np.zeros_like(data)
        
        z_scores = np.abs((data - mean) / std)
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        return anomaly_indices, z_scores
    
    @staticmethod
    def iqr_detection(
        data: np.ndarray,
        multiplier: float = 1.5
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        More robust to outliers than Z-score. Works well for skewed data.
        
        Args:
            data: 1D array of values
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme)
            
        Returns:
            Tuple of (anomaly_indices, (lower_bound, upper_bound))
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)
        
        anomaly_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        
        return anomaly_indices, (lower_bound, upper_bound)
    
    @staticmethod
    def isolation_forest_detection(
        data: np.ndarray,
        contamination: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Works well with high-dimensional data. Doesn't assume any distribution.
        Based on the principle that anomalies are "few and different" -
        they're easier to isolate with random partitioning.
        
        Args:
            data: Array of shape (n_samples, n_features) or (n_samples,)
            contamination: Expected proportion of anomalies
            
        Returns:
            Tuple of (anomaly_indices, anomaly_scores)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Standardize for better performance
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = model.fit_predict(data_scaled)
        scores = model.decision_function(data_scaled)
        
        # predictions: 1 for normal, -1 for anomaly
        anomaly_indices = np.where(predictions == -1)[0]
        
        return anomaly_indices, scores
    
    @staticmethod
    def embedding_anomaly_detection(
        embeddings: np.ndarray,
        contamination: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalous logs using their embeddings.
        
        Logs that are semantically different from the majority will have
        embeddings that are distant from the main clusters.
        
        Args:
            embeddings: Array of shape (n_logs, embedding_dim)
            contamination: Expected proportion of anomalies
            
        Returns:
            Tuple of (anomaly_indices, anomaly_scores)
        """
        return AnomalyDetector.isolation_forest_detection(embeddings, contamination)


def demo_anomaly_detection():
    """Demonstrate anomaly detection methods."""
    print("\n" + "="*70)
    print("SECTION 4: ANOMALY DETECTION DEMONSTRATION")
    print("="*70)
    
    detector = AnomalyDetector()
    
    # Use sample metrics
    cpu_data = np.array(SAMPLE_METRICS["cpu_percent"])
    timestamps = SAMPLE_METRICS["timestamps"]
    
    print("\n📊 CPU Utilization Data:")
    for ts, cpu in zip(timestamps, cpu_data):
        print(f"   {ts}: {cpu}%")
    
    # Z-score detection
    print("\n--- Z-Score Anomaly Detection ---")
    anomalies, z_scores = detector.zscore_detection(cpu_data, threshold=2.0)
    print(f"Anomalies detected at indices: {anomalies}")
    for idx in anomalies:
        print(f"   {timestamps[idx]}: {cpu_data[idx]}% (z-score: {z_scores[idx]:.2f})")
    
    # IQR detection
    print("\n--- IQR Anomaly Detection ---")
    anomalies, bounds = detector.iqr_detection(cpu_data)
    print(f"Normal range: {bounds[0]:.1f}% - {bounds[1]:.1f}%")
    print(f"Anomalies detected at indices: {anomalies}")
    for idx in anomalies:
        print(f"   {timestamps[idx]}: {cpu_data[idx]}%")
    
    # Isolation Forest (multivariate)
    print("\n--- Isolation Forest (Multivariate) ---")
    multivariate_data = np.column_stack([
        SAMPLE_METRICS["cpu_percent"],
        SAMPLE_METRICS["memory_percent"],
        SAMPLE_METRICS["request_latency_ms"]
    ])
    anomalies, scores = detector.isolation_forest_detection(
        multivariate_data,
        contamination=0.2
    )
    print(f"Anomalies detected at indices: {anomalies}")
    for idx in anomalies:
        print(f"   {timestamps[idx]}: CPU={multivariate_data[idx][0]}%, "
              f"MEM={multivariate_data[idx][1]}%, "
              f"Latency={multivariate_data[idx][2]}ms")
    
    return detector


# =============================================================================
# SECTION 5: CLASSIFICATION AND CLUSTERING
# =============================================================================

class LogClassifier:
    """
    Classification and clustering for log analysis.
    
    - Clustering: Automatically discover log categories
    - Classification: Assign logs to predefined categories
    """
    
    def __init__(self, embedder: BedrockEmbeddings):
        self.embedder = embedder
    
    def cluster_logs(
        self,
        logs: List[str],
        n_clusters: int = None,
        method: str = "kmeans"
    ) -> Dict:
        """
        Cluster logs into groups based on semantic similarity.
        
        This automatically discovers categories in your logs without
        requiring predefined labels.
        
        Args:
            logs: List of log messages
            n_clusters: Number of clusters (auto-detected for DBSCAN)
            method: "kmeans" or "dbscan"
            
        Returns:
            Dict with cluster assignments and cluster info
        """
        # Generate embeddings
        print(f"Generating embeddings for {len(logs)} logs...")
        embeddings = self.embedder.embed_batch(logs)
        
        if method == "kmeans":
            if n_clusters is None:
                n_clusters = min(5, len(logs) // 2)
            
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(embeddings)
            
        elif method == "dbscan":
            # DBSCAN automatically determines number of clusters
            # eps and min_samples may need tuning for your data
            model = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
            labels = model.fit_predict(embeddings)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Organize results
        clusters = {}
        for i, (log, label) in enumerate(zip(logs, labels)):
            label_key = int(label)
            if label_key not in clusters:
                clusters[label_key] = []
            clusters[label_key].append({"index": i, "text": log})
        
        return {
            "labels": labels.tolist(),
            "clusters": clusters,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "embeddings": embeddings
        }
    
    def classify_log(
        self,
        log: str,
        categories: Dict[str, List[str]],
        threshold: float = 0.5
    ) -> Tuple[str, float]:
        """
        Classify a log into predefined categories using embedding similarity.
        
        Args:
            log: The log message to classify
            categories: Dict mapping category names to example logs
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (category_name, confidence_score)
        """
        log_embedding = self.embedder.embed_text(log)
        
        best_category = "unknown"
        best_score = 0.0
        
        for category, examples in categories.items():
            # Get embeddings for category examples
            example_embeddings = self.embedder.embed_batch(examples)
            
            # Calculate similarity to each example
            similarities = cosine_similarity(
                log_embedding.reshape(1, -1),
                example_embeddings
            )[0]
            
            # Use max similarity as category score
            category_score = np.max(similarities)
            
            if category_score > best_score:
                best_score = category_score
                best_category = category
        
        if best_score < threshold:
            best_category = "unknown"
        
        return best_category, best_score


def demo_classification_clustering(embedder: BedrockEmbeddings):
    """Demonstrate classification and clustering."""
    print("\n" + "="*70)
    print("SECTION 5: CLASSIFICATION AND CLUSTERING DEMONSTRATION")
    print("="*70)
    
    classifier = LogClassifier(embedder)
    
    # Clustering demonstration
    print("\n--- Automatic Log Clustering ---")
    result = classifier.cluster_logs(SAMPLE_LOGS, n_clusters=5, method="kmeans")
    
    print(f"\nDiscovered {result['n_clusters']} clusters:")
    for cluster_id, logs in sorted(result["clusters"].items()):
        print(f"\n📁 Cluster {cluster_id} ({len(logs)} logs):")
        for log_info in logs[:2]:  # Show first 2 logs per cluster
            print(f"   • {log_info['text'][:60]}...")
    
    # Classification demonstration
    print("\n--- Log Classification ---")
    
    # Define categories with example logs
    categories = {
        "authentication": [
            "Login failed for user - invalid password",
            "Authentication token expired",
            "User permission denied"
        ],
        "database": [
            "Database connection refused",
            "Query timeout exceeded",
            "Connection pool exhausted"
        ],
        "memory": [
            "OutOfMemoryError occurred",
            "Heap space exceeded",
            "GC overhead limit reached"
        ],
        "network": [
            "Connection timeout to service",
            "Socket timeout - no response",
            "High latency detected"
        ]
    }
    
    # Classify some test logs
    test_logs = [
        "2024-01-15 11:00:00 ERROR: Unable to authenticate user - credentials rejected",
        "2024-01-15 11:00:01 ERROR: PostgreSQL connection pool depleted",
        "2024-01-15 11:00:02 ERROR: Service unreachable after 5000ms",
    ]
    
    print("\nClassifying new logs:")
    for log in test_logs:
        category, confidence = classifier.classify_log(log, categories)
        print(f"\n📝 Log: '{log[:50]}...'")
        print(f"   Category: {category} (confidence: {confidence:.3f})")
    
    return classifier


# =============================================================================
# SECTION 6: PUTTING IT ALL TOGETHER
# =============================================================================

def demo_integrated_pipeline():
    """
    Demonstrate an integrated observability pipeline.
    
    This shows how the building blocks combine:
    1. Logs are embedded and stored
    2. Anomaly detection identifies unusual patterns
    3. Similar logs are retrieved for context
    4. LLM synthesizes findings into actionable insights
    """
    print("\n" + "="*70)
    print("SECTION 6: INTEGRATED PIPELINE DEMONSTRATION")
    print("="*70)
    
    # Initialize components
    embedder = BedrockEmbeddings()
    vector_store = SimpleVectorStore(embedder)
    detector = AnomalyDetector()
    llm = BedrockLLM()
    
    # Step 1: Index logs
    print("\n📥 Step 1: Indexing logs into vector store...")
    for i, log in enumerate(SAMPLE_LOGS):
        level = "INFO"
        for lvl in ["CRITICAL", "ERROR", "WARN"]:
            if lvl in log:
                level = lvl
                break
        vector_store.add(f"log_{i}", log, {"level": level})
    print(f"   Indexed {len(SAMPLE_LOGS)} logs")
    
    # Step 2: Detect metric anomalies
    print("\n📈 Step 2: Detecting metric anomalies...")
    metrics_data = np.column_stack([
        SAMPLE_METRICS["cpu_percent"],
        SAMPLE_METRICS["memory_percent"],
        SAMPLE_METRICS["request_latency_ms"]
    ])
    anomaly_indices, _ = detector.isolation_forest_detection(
        metrics_data,
        contamination=0.2
    )
    print(f"   Detected anomalies at time indices: {anomaly_indices}")
    
    # Step 3: Find related logs (semantic search)
    print("\n🔍 Step 3: Finding related logs via semantic search...")
    search_results = vector_store.search("high resource usage performance degradation", top_k=5)
    related_logs = [record.text for record, _ in search_results]
    print(f"   Found {len(related_logs)} related logs")
    
    # Step 4: LLM analysis
    print("\n🤖 Step 4: Generating AI-powered analysis...")
    
    # Prepare context for LLM
    anomaly_times = [SAMPLE_METRICS["timestamps"][i] for i in anomaly_indices]
    anomaly_metrics = {
        "anomalous_timestamps": anomaly_times,
        "metrics_at_anomaly": {
            ts: {
                "cpu": SAMPLE_METRICS["cpu_percent"][i],
                "memory": SAMPLE_METRICS["memory_percent"][i],
                "latency_ms": SAMPLE_METRICS["request_latency_ms"][i]
            }
            for i, ts in zip(anomaly_indices, anomaly_times)
        }
    }
    
    analysis = llm.suggest_root_cause(related_logs, anomaly_metrics)
    
    print("\n" + "="*70)
    print("📊 INTEGRATED ANALYSIS RESULTS")
    print("="*70)
    print(f"\n{analysis}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run all demonstrations.
    
    Note: These demos make API calls to Amazon Bedrock.
    Ensure you have:
    1. AWS credentials configured
    2. Bedrock model access enabled for:
       - amazon.titan-embed-text-v2:0
       - anthropic.claude-3-sonnet-20240229-v1:0
    """
    print("="*70)
    print("CHAPTER 3: BUILDING BLOCKS — CODE EXAMPLES")
    print("AI-Powered Observability with Amazon Bedrock")
    print("="*70)
    
    try:
        # Run individual demos
        embedder, _ = demo_embeddings()
        demo_vector_search(embedder)
        demo_llm_capabilities()
        demo_anomaly_detection()
        demo_classification_clustering(embedder)
        
        # Run integrated demo
        demo_integrated_pipeline()
        
        print("\n" + "="*70)
        print("✅ All demonstrations completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure AWS credentials are configured (aws configure)")
        print("2. Enable Bedrock models in AWS Console")
        print("3. Check your AWS region supports Bedrock")
        raise


if __name__ == "__main__":
    main()
