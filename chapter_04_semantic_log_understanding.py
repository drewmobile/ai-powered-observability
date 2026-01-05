"""
Chapter 4: Semantic Log Understanding
======================================

Code examples demonstrating semantic log analysis capabilities:
- Semantic search (find logs by meaning, not keywords)
- Similar log discovery
- Entity extraction from logs
- Natural language querying
- Log summarization
- Semantic error classification
- Novelty detection

Prerequisites:
    pip install boto3 numpy scikit-learn

AWS Configuration:
    Ensure you have AWS credentials configured with access to Amazon Bedrock.
    Enable the following models in your Bedrock console:
    - amazon.titan-embed-text-v2:0 (for embeddings)
    - anthropic.claude-3-sonnet-20240229-v1:0 (for LLM tasks)

This chapter builds on Chapter 3's building blocks to create practical
log analysis capabilities.
"""

import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import boto3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"


def get_bedrock_client(region_name: str = "us-east-1"):
    """Create a Bedrock runtime client."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name
    )


# =============================================================================
# SAMPLE LOG DATA - More realistic and varied
# =============================================================================

SAMPLE_LOGS = [
    # Authentication / Authorization issues (various phrasings)
    {
        "timestamp": "2024-01-15T10:23:45.123Z",
        "level": "ERROR",
        "service": "auth-service",
        "message": "Login failed for user john@example.com - invalid password after 3 attempts"
    },
    {
        "timestamp": "2024-01-15T10:23:46.456Z",
        "level": "ERROR",
        "service": "auth-service",
        "message": "Authentication rejected: token expired for session sess_abc123"
    },
    {
        "timestamp": "2024-01-15T10:23:47.789Z",
        "level": "WARN",
        "service": "auth-service",
        "message": "Access denied - user alice lacks required role: billing_admin"
    },
    {
        "timestamp": "2024-01-15T10:23:48.012Z",
        "level": "ERROR",
        "service": "api-gateway",
        "message": "Unauthorized request to /api/v2/payments - missing bearer token"
    },
    {
        "timestamp": "2024-01-15T10:23:49.345Z",
        "level": "ERROR",
        "service": "auth-service",
        "message": "OAuth callback failed: invalid state parameter from provider google"
    },
    
    # Database issues (various phrasings)
    {
        "timestamp": "2024-01-15T10:24:00.123Z",
        "level": "ERROR",
        "service": "order-service",
        "message": "Connection refused to postgres://db-primary.internal:5432/orders"
    },
    {
        "timestamp": "2024-01-15T10:24:01.456Z",
        "level": "ERROR",
        "service": "inventory-service",
        "message": "Query timeout after 30000ms: SELECT * FROM products WHERE category_id = 42"
    },
    {
        "timestamp": "2024-01-15T10:24:02.789Z",
        "level": "WARN",
        "service": "user-service",
        "message": "Database connection pool exhausted - 50/50 connections in use, 12 requests waiting"
    },
    {
        "timestamp": "2024-01-15T10:24:03.012Z",
        "level": "ERROR",
        "service": "payment-service",
        "message": "Deadlock detected in transaction txn_789xyz - rolling back"
    },
    {
        "timestamp": "2024-01-15T10:24:04.345Z",
        "level": "ERROR",
        "service": "reporting-service",
        "message": "Unable to establish database connection: SQLSTATE[HY000] [2002] Connection timed out"
    },
    
    # Memory / Resource issues
    {
        "timestamp": "2024-01-15T10:25:00.123Z",
        "level": "CRITICAL",
        "service": "order-service",
        "message": "java.lang.OutOfMemoryError: Java heap space - current heap 7.8GB/8GB"
    },
    {
        "timestamp": "2024-01-15T10:25:01.456Z",
        "level": "WARN",
        "service": "order-service",
        "message": "GC overhead limit exceeded - 98% of time spent in garbage collection"
    },
    {
        "timestamp": "2024-01-15T10:25:02.789Z",
        "level": "ERROR",
        "service": "image-processor",
        "message": "Memory allocation failed: unable to allocate 512MB buffer for image resize"
    },
    {
        "timestamp": "2024-01-15T10:25:03.012Z",
        "level": "WARN",
        "service": "cache-service",
        "message": "Redis memory usage at 95% - evicting keys using LRU policy"
    },
    
    # Network / Connectivity issues
    {
        "timestamp": "2024-01-15T10:26:00.123Z",
        "level": "ERROR",
        "service": "api-gateway",
        "message": "Connection timeout after 5000ms to payment-service.prod.internal:8080"
    },
    {
        "timestamp": "2024-01-15T10:26:01.456Z",
        "level": "ERROR",
        "service": "notification-service",
        "message": "Socket timeout: no response from email-relay.internal within 10s"
    },
    {
        "timestamp": "2024-01-15T10:26:02.789Z",
        "level": "WARN",
        "service": "api-gateway",
        "message": "High latency detected: 2500ms response time from inventory-service (p99: 200ms)"
    },
    {
        "timestamp": "2024-01-15T10:26:03.012Z",
        "level": "ERROR",
        "service": "shipping-service",
        "message": "DNS resolution failed for external-carrier-api.logistics.com"
    },
    {
        "timestamp": "2024-01-15T10:26:04.345Z",
        "level": "ERROR",
        "service": "webhook-service",
        "message": "HTTP 503 Service Unavailable from https://partner-api.example.com/callbacks"
    },
    
    # Application errors
    {
        "timestamp": "2024-01-15T10:27:00.123Z",
        "level": "ERROR",
        "service": "order-service",
        "message": "NullPointerException in OrderProcessor.processPayment() at line 234"
    },
    {
        "timestamp": "2024-01-15T10:27:01.456Z",
        "level": "ERROR",
        "service": "inventory-service",
        "message": "ValidationError: negative quantity -5 for SKU PROD-12345"
    },
    {
        "timestamp": "2024-01-15T10:27:02.789Z",
        "level": "ERROR",
        "service": "pricing-service",
        "message": "Division by zero in discount calculation for campaign SUMMER2024"
    },
    {
        "timestamp": "2024-01-15T10:27:03.012Z",
        "level": "ERROR",
        "service": "checkout-service",
        "message": "Invalid state transition: cannot move order ORD-98765 from CANCELLED to SHIPPED"
    },
    
    # Success / Info messages (for contrast)
    {
        "timestamp": "2024-01-15T10:28:00.123Z",
        "level": "INFO",
        "service": "auth-service",
        "message": "User bob@example.com successfully logged in from IP 192.168.1.50"
    },
    {
        "timestamp": "2024-01-15T10:28:01.456Z",
        "level": "INFO",
        "service": "order-service",
        "message": "Order ORD-11111 processed successfully - total $149.99, 3 items"
    },
    {
        "timestamp": "2024-01-15T10:28:02.789Z",
        "level": "INFO",
        "service": "notification-service",
        "message": "Email sent to customer@example.com - order confirmation for ORD-11111"
    },
    {
        "timestamp": "2024-01-15T10:28:03.012Z",
        "level": "INFO",
        "service": "deployment-service",
        "message": "Deployment completed: order-service v2.4.1 rolled out to 5/5 pods"
    },
    {
        "timestamp": "2024-01-15T10:28:04.345Z",
        "level": "DEBUG",
        "service": "cache-service",
        "message": "Cache hit for key user:profile:12345 - response time 2ms"
    },
]


# =============================================================================
# CORE CLASSES
# =============================================================================

class BedrockEmbeddings:
    """Generate embeddings using Amazon Bedrock Titan model."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.client = get_bedrock_client(region_name)
        self.model_id = EMBEDDING_MODEL_ID
        self.embedding_dimension = 1024
        self._cache = {}  # Simple cache to avoid re-embedding identical text
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text, with caching."""
        if text in self._cache:
            return self._cache[text]
        
        body = json.dumps({
            "inputText": text,
            "dimensions": self.embedding_dimension,
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
        """Generate embeddings for multiple texts."""
        embeddings = [self.embed_text(text) for text in texts]
        return np.array(embeddings)


class BedrockLLM:
    """Interact with Claude via Amazon Bedrock."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.client = get_bedrock_client(region_name)
        self.model_id = LLM_MODEL_ID
    
    def invoke(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.3
    ) -> str:
        """Send a prompt to Claude and get a response."""
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


# =============================================================================
# SECTION 1: SEMANTIC LOG SEARCH
# =============================================================================

@dataclass
class LogEntry:
    """Structured log entry with embedding."""
    id: str
    timestamp: str
    level: str
    service: str
    message: str
    raw: str
    embedding: np.ndarray = None
    metadata: Dict = field(default_factory=dict)


class SemanticLogStore:
    """
    A semantic log store that enables meaning-based search.
    
    Unlike traditional keyword search, semantic search finds logs based on
    meaning. "authentication failures" finds logs about "login denied",
    "invalid password", "token expired" - even without shared keywords.
    """
    
    def __init__(self, embedder: BedrockEmbeddings):
        self.embedder = embedder
        self.logs: List[LogEntry] = []
    
    def ingest(self, log_data: Dict) -> LogEntry:
        """
        Ingest a single log entry, generating its embedding.
        
        The embedding captures the semantic content of the log message,
        enabling similarity-based search.
        """
        # Create the raw log string for embedding
        raw = f"{log_data['level']} {log_data['service']}: {log_data['message']}"
        
        # Generate embedding
        embedding = self.embedder.embed_text(raw)
        
        # Create log entry
        entry = LogEntry(
            id=f"log_{len(self.logs)}",
            timestamp=log_data["timestamp"],
            level=log_data["level"],
            service=log_data["service"],
            message=log_data["message"],
            raw=raw,
            embedding=embedding
        )
        
        self.logs.append(entry)
        return entry
    
    def ingest_batch(self, logs_data: List[Dict]) -> List[LogEntry]:
        """Ingest multiple log entries."""
        return [self.ingest(log) for log in logs_data]
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        level_filter: str = None,
        service_filter: str = None,
        min_similarity: float = 0.0
    ) -> List[Tuple[LogEntry, float]]:
        """
        Search logs by semantic similarity to a query.
        
        This is the core of semantic log understanding. Instead of
        exact keyword matching, we find logs whose meaning is similar
        to the query.
        
        Args:
            query: Natural language query (e.g., "authentication failures")
            top_k: Number of results to return
            level_filter: Optional filter by log level
            service_filter: Optional filter by service name
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (LogEntry, similarity_score) tuples
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)
        
        # Calculate similarity to all logs
        results = []
        for log in self.logs:
            # Apply filters
            if level_filter and log.level != level_filter:
                continue
            if service_filter and log.service != service_filter:
                continue
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                log.embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= min_similarity:
                results.append((log, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def find_similar(
        self,
        log_id: str,
        top_k: int = 5,
        exclude_same_service: bool = False
    ) -> List[Tuple[LogEntry, float]]:
        """
        Find logs similar to a specific log entry.
        
        Useful for:
        - Finding related errors during investigation
        - Discovering patterns across services
        - Identifying recurring issues
        """
        target = next((log for log in self.logs if log.id == log_id), None)
        if not target:
            raise ValueError(f"Log {log_id} not found")
        
        results = []
        for log in self.logs:
            if log.id == log_id:
                continue
            if exclude_same_service and log.service == target.service:
                continue
            
            similarity = cosine_similarity(
                target.embedding.reshape(1, -1),
                log.embedding.reshape(1, -1)
            )[0][0]
            results.append((log, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_by_level(self, level: str) -> List[LogEntry]:
        """Get all logs of a specific level."""
        return [log for log in self.logs if log.level == level]
    
    def get_by_service(self, service: str) -> List[LogEntry]:
        """Get all logs from a specific service."""
        return [log for log in self.logs if log.service == service]


def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    print("\n" + "="*70)
    print("SECTION 1: SEMANTIC LOG SEARCH")
    print("="*70)
    
    embedder = BedrockEmbeddings()
    store = SemanticLogStore(embedder)
    
    # Ingest sample logs
    print("\n📥 Ingesting sample logs...")
    store.ingest_batch(SAMPLE_LOGS)
    print(f"   Indexed {len(store.logs)} log entries")
    
    # Demonstrate semantic search with various queries
    queries = [
        ("authentication failures", "Finds login errors, token issues, access denied"),
        ("database connectivity problems", "Finds connection refused, timeouts, pool exhaustion"),
        ("system running out of memory", "Finds OOM errors, heap issues, allocation failures"),
        ("slow response times", "Finds latency warnings, timeouts"),
        ("successful operations", "Finds INFO logs about completed tasks"),
    ]
    
    print("\n--- Semantic Search Examples ---")
    for query, description in queries:
        print(f"\n🔍 Query: \"{query}\"")
        print(f"   ({description})")
        results = store.semantic_search(query, top_k=3)
        for log, score in results:
            print(f"   [{score:.3f}] [{log.level}] {log.service}: {log.message[:60]}...")
    
    # Compare with what keyword search would miss
    print("\n--- Semantic vs Keyword Search ---")
    print("\nQuery: 'login problems'")
    print("A keyword search for 'login' would miss these semantically related logs:")
    results = store.semantic_search("login problems", top_k=5)
    for log, score in results:
        has_keyword = "login" in log.message.lower()
        marker = "✓ has 'login'" if has_keyword else "✗ no 'login' keyword"
        print(f"   [{score:.3f}] {marker}")
        print(f"            {log.message[:65]}...")
    
    return store


# =============================================================================
# SECTION 2: ENTITY EXTRACTION
# =============================================================================

@dataclass
class ExtractedEntities:
    """Entities extracted from a log message."""
    services: List[str] = field(default_factory=list)
    hosts: List[str] = field(default_factory=list)
    ips: List[str] = field(default_factory=list)
    users: List[str] = field(default_factory=list)
    request_ids: List[str] = field(default_factory=list)
    order_ids: List[str] = field(default_factory=list)
    error_codes: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    durations: List[str] = field(default_factory=list)
    numeric_values: List[Tuple[str, str]] = field(default_factory=list)  # (value, context)


class EntityExtractor:
    """
    Extract meaningful entities from log messages.
    
    Entity extraction identifies the "who, what, where" in logs:
    - Service names, hostnames, IP addresses
    - User IDs, request IDs, order IDs
    - URLs, error codes, durations
    
    This can be done with regex patterns (fast, limited) or
    LLMs (slower, more accurate, handles novel formats).
    """
    
    def __init__(self, llm: BedrockLLM = None):
        self.llm = llm
        
        # Regex patterns for common entities
        self.patterns = {
            "ip": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
            "host": r'\b[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)+(?::\d+)?\b',
            "duration_ms": r'\b(\d+)\s*(?:ms|milliseconds?)\b',
            "duration_s": r'\b(\d+(?:\.\d+)?)\s*(?:s|seconds?)\b',
            "request_id": r'\b(?:req|request|sess|session|txn|transaction)[_-]?[a-zA-Z0-9]{6,}\b',
            "order_id": r'\b(?:ORD|order)[_-]?\d{4,}\b',
            "error_code": r'\b(?:SQLSTATE|HTTP|E[A-Z]+)\[?[A-Z0-9]+\]?\b',
            "percentage": r'\b(\d+(?:\.\d+)?)\s*%',
            "memory_size": r'\b(\d+(?:\.\d+)?)\s*(?:GB|MB|KB|TB)\b',
        }
    
    def extract_with_regex(self, message: str) -> ExtractedEntities:
        """
        Extract entities using regex patterns.
        
        Fast but limited to predefined patterns.
        """
        entities = ExtractedEntities()
        
        # Extract IPs
        entities.ips = re.findall(self.patterns["ip"], message)
        
        # Extract emails (as users)
        entities.users = re.findall(self.patterns["email"], message)
        
        # Extract URLs
        entities.urls = re.findall(self.patterns["url"], message)
        
        # Extract hostnames
        entities.hosts = re.findall(self.patterns["host"], message)
        # Filter out emails from hosts
        entities.hosts = [h for h in entities.hosts if "@" not in h and h not in entities.urls]
        
        # Extract request/session/transaction IDs
        entities.request_ids = re.findall(self.patterns["request_id"], message, re.IGNORECASE)
        
        # Extract order IDs
        entities.order_ids = re.findall(self.patterns["order_id"], message, re.IGNORECASE)
        
        # Extract error codes
        entities.error_codes = re.findall(self.patterns["error_code"], message)
        
        # Extract durations
        ms_durations = re.findall(self.patterns["duration_ms"], message)
        entities.durations = [f"{d}ms" for d in ms_durations]
        s_durations = re.findall(self.patterns["duration_s"], message)
        entities.durations.extend([f"{d}s" for d in s_durations])
        
        # Extract numeric values with context
        percentages = re.findall(self.patterns["percentage"], message)
        for p in percentages:
            entities.numeric_values.append((f"{p}%", "percentage"))
        
        memory = re.findall(self.patterns["memory_size"], message)
        for m in memory:
            entities.numeric_values.append((m, "memory"))
        
        return entities
    
    def extract_with_llm(self, message: str, context: str = None) -> Dict:
        """
        Extract entities using LLM for more accurate, context-aware extraction.
        
        Slower but handles novel formats and understands context.
        """
        if not self.llm:
            raise ValueError("LLM not configured for entity extraction")
        
        system_prompt = """You are an expert at extracting entities from system logs.
        Extract all meaningful entities and return them as JSON.
        Be precise and only extract entities that are clearly present."""
        
        prompt = f"""Extract entities from this log message:

"{message}"

Return a JSON object with these fields (use empty arrays if not found):
- services: service/application names
- hosts: hostnames or server names  
- ips: IP addresses
- users: user identifiers (emails, usernames)
- request_ids: request/session/transaction IDs
- error_codes: error codes or status codes
- urls: URLs or endpoints
- durations: time durations
- other_ids: any other identifiable IDs

Return only valid JSON, no explanation."""

        response = self.llm.invoke(prompt, system_prompt, temperature=0.1)
        
        # Parse JSON response
        try:
            # Clean up response if needed
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response", "raw": response}


def demo_entity_extraction():
    """Demonstrate entity extraction from logs."""
    print("\n" + "="*70)
    print("SECTION 2: ENTITY EXTRACTION")
    print("="*70)
    
    extractor = EntityExtractor()
    
    # Test logs with various entities
    test_logs = [
        "Connection timeout after 5000ms to payment-service.prod.internal:8080",
        "User john@example.com logged in from IP 192.168.1.50",
        "Query timeout after 30000ms: SELECT * FROM products WHERE category_id = 42",
        "Order ORD-98765 failed - transaction txn_abc123 rolled back",
        "HTTP 503 from https://partner-api.example.com/callbacks - retry in 30s",
        "java.lang.OutOfMemoryError: Java heap space - current heap 7.8GB/8GB",
    ]
    
    print("\n--- Regex-Based Entity Extraction ---")
    for log in test_logs:
        print(f"\n📝 Log: \"{log[:60]}...\"")
        entities = extractor.extract_with_regex(log)
        
        # Print non-empty entity lists
        if entities.hosts:
            print(f"   🖥️  Hosts: {entities.hosts}")
        if entities.ips:
            print(f"   🌐 IPs: {entities.ips}")
        if entities.users:
            print(f"   👤 Users: {entities.users}")
        if entities.request_ids:
            print(f"   🔑 Request IDs: {entities.request_ids}")
        if entities.order_ids:
            print(f"   📦 Order IDs: {entities.order_ids}")
        if entities.urls:
            print(f"   🔗 URLs: {entities.urls}")
        if entities.durations:
            print(f"   ⏱️  Durations: {entities.durations}")
        if entities.error_codes:
            print(f"   ⚠️  Error codes: {entities.error_codes}")
        if entities.numeric_values:
            print(f"   🔢 Values: {entities.numeric_values}")
    
    return extractor


# =============================================================================
# SECTION 3: NATURAL LANGUAGE QUERYING
# =============================================================================

class NaturalLanguageLogQuery:
    """
    Translate natural language questions into log queries.
    
    This enables non-technical users to search logs without knowing
    query syntax, and helps everyone search faster by speaking naturally.
    """
    
    def __init__(self, llm: BedrockLLM, log_store: SemanticLogStore):
        self.llm = llm
        self.log_store = log_store
    
    def interpret_query(self, natural_query: str) -> Dict:
        """
        Interpret a natural language query into structured parameters.
        
        Returns a dict with:
        - semantic_query: The search terms for embedding-based search
        - filters: Any filters to apply (level, service, time range)
        - intent: What the user is trying to do
        """
        system_prompt = """You are an expert at interpreting natural language queries about system logs.
        Analyze the query and extract:
        1. The semantic search terms (what kind of logs to find)
        2. Any filters (log level, service name, time range)
        3. The user's intent (search, count, summarize, investigate)
        
        Return JSON only."""
        
        prompt = f"""Interpret this log query:

"{natural_query}"

Return JSON with:
- semantic_query: string - key terms for semantic search
- level_filter: string or null - log level filter (ERROR, WARN, INFO, etc.)
- service_filter: string or null - service name filter
- time_filter: object or null - with "start" and "end" if time mentioned
- intent: string - one of: search, count, summarize, investigate, compare
- explanation: string - brief explanation of interpretation

Return only valid JSON."""

        response = self.llm.invoke(prompt, system_prompt, temperature=0.1)
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "semantic_query": natural_query,
                "level_filter": None,
                "service_filter": None,
                "intent": "search",
                "explanation": "Fallback to direct search"
            }
    
    def execute_query(self, natural_query: str, top_k: int = 10) -> Dict:
        """
        Execute a natural language query against the log store.
        
        This combines query interpretation with semantic search.
        """
        # Interpret the query
        interpretation = self.interpret_query(natural_query)
        
        # Execute semantic search with interpreted parameters
        results = self.log_store.semantic_search(
            query=interpretation.get("semantic_query", natural_query),
            top_k=top_k,
            level_filter=interpretation.get("level_filter"),
            service_filter=interpretation.get("service_filter")
        )
        
        return {
            "interpretation": interpretation,
            "results": results,
            "result_count": len(results)
        }
    
    def conversational_search(self, queries: List[str]) -> List[Dict]:
        """
        Handle a conversation of queries, with context from previous queries.
        
        This enables iterative refinement: "Show me errors" -> "Only from order-service"
        """
        results_history = []
        context = ""
        
        for query in queries:
            # Add context from previous queries
            full_query = f"{context} {query}".strip() if context else query
            
            result = self.execute_query(full_query)
            results_history.append({
                "query": query,
                "full_query": full_query,
                "result": result
            })
            
            # Update context for next query
            context = f"Previous query was about: {result['interpretation'].get('semantic_query', query)}"
        
        return results_history


def demo_natural_language_query(log_store: SemanticLogStore):
    """Demonstrate natural language querying."""
    print("\n" + "="*70)
    print("SECTION 3: NATURAL LANGUAGE QUERYING")
    print("="*70)
    
    llm = BedrockLLM()
    nl_query = NaturalLanguageLogQuery(llm, log_store)
    
    # Example natural language queries
    queries = [
        "Show me all authentication failures",
        "What database errors happened?",
        "Find errors from the order service",
        "Any memory problems?",
        "Show successful operations",
    ]
    
    print("\n--- Natural Language Query Examples ---")
    for query in queries:
        print(f"\n💬 \"{query}\"")
        result = nl_query.execute_query(query, top_k=3)
        
        interp = result["interpretation"]
        print(f"   📋 Interpreted as: {interp.get('semantic_query', 'N/A')}")
        if interp.get("level_filter"):
            print(f"   🏷️  Level filter: {interp['level_filter']}")
        if interp.get("service_filter"):
            print(f"   🔧 Service filter: {interp['service_filter']}")
        print(f"   🎯 Intent: {interp.get('intent', 'search')}")
        
        print(f"   📊 Results ({result['result_count']} found):")
        for log, score in result["results"][:3]:
            print(f"      [{score:.3f}] {log.message[:55]}...")
    
    return nl_query


# =============================================================================
# SECTION 4: LOG SUMMARIZATION
# =============================================================================

class LogSummarizer:
    """
    Summarize log entries into human-readable narratives.
    
    Types of summaries:
    - Timeline: Chronological narrative of events
    - Error: Grouped and explained errors
    - Impact: User-visible effects
    - Root cause: Synthesized hypothesis
    """
    
    def __init__(self, llm: BedrockLLM):
        self.llm = llm
    
    def timeline_summary(self, logs: List[LogEntry]) -> str:
        """
        Create a chronological narrative of events.
        
        Useful for incident timelines and post-mortems.
        """
        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)
        
        # Format logs for the prompt
        log_text = "\n".join([
            f"[{log.timestamp}] [{log.level}] {log.service}: {log.message}"
            for log in sorted_logs
        ])
        
        system_prompt = """You are an SRE creating an incident timeline.
        Create a clear, chronological narrative that:
        1. Highlights key events and their timing
        2. Shows the progression of the issue
        3. Notes any patterns or correlations
        Keep it concise but complete."""
        
        prompt = f"""Create a timeline summary of these log events:

{log_text}

Write a brief chronological narrative explaining what happened."""

        return self.llm.invoke(prompt, system_prompt)
    
    def error_summary(self, logs: List[LogEntry]) -> str:
        """
        Summarize and categorize errors.
        
        Groups errors by type and provides counts and explanations.
        """
        # Filter to error logs only
        error_logs = [log for log in logs if log.level in ["ERROR", "CRITICAL"]]
        
        if not error_logs:
            return "No errors found in the provided logs."
        
        log_text = "\n".join([
            f"[{log.level}] {log.service}: {log.message}"
            for log in error_logs
        ])
        
        system_prompt = """You are an SRE analyzing system errors.
        Categorize and summarize the errors:
        1. Group by error type
        2. Count occurrences of each type
        3. Explain what each error type means
        4. Note any patterns or correlations"""
        
        prompt = f"""Analyze and summarize these errors:

{log_text}

Provide a categorized summary with counts and explanations."""

        return self.llm.invoke(prompt, system_prompt)
    
    def impact_summary(self, logs: List[LogEntry]) -> str:
        """
        Summarize the user-visible impact of issues.
        
        Focuses on what users experienced, not technical details.
        """
        log_text = "\n".join([
            f"[{log.timestamp}] [{log.level}] {log.service}: {log.message}"
            for log in logs
        ])
        
        system_prompt = """You are explaining a system issue to stakeholders.
        Focus on user impact:
        1. What functionality was affected?
        2. How many users might have been impacted?
        3. What did users experience (errors, slowness, etc.)?
        4. How long did the impact last?
        Avoid technical jargon. Be concise."""
        
        prompt = f"""Based on these logs, summarize the user impact:

{log_text}

Focus on what users experienced, not technical details."""

        return self.llm.invoke(prompt, system_prompt)
    
    def root_cause_summary(
        self,
        logs: List[LogEntry],
        metrics: Dict = None
    ) -> str:
        """
        Synthesize evidence into a root cause hypothesis.
        
        Combines log analysis with optional metrics for deeper insight.
        """
        log_text = "\n".join([
            f"[{log.timestamp}] [{log.level}] {log.service}: {log.message}"
            for log in logs
        ])
        
        metrics_text = ""
        if metrics:
            metrics_text = f"\n\nRelevant metrics:\n{json.dumps(metrics, indent=2)}"
        
        system_prompt = """You are an expert SRE performing root cause analysis.
        Analyze the evidence and:
        1. Identify the most likely root cause
        2. Explain the chain of events
        3. Note what evidence supports your hypothesis
        4. Suggest what to investigate next
        Be specific and evidence-based."""
        
        prompt = f"""Analyze these logs to determine root cause:

{log_text}
{metrics_text}

What is the most likely root cause? Explain your reasoning."""

        return self.llm.invoke(prompt, system_prompt)


def demo_log_summarization(log_store: SemanticLogStore):
    """Demonstrate log summarization capabilities."""
    print("\n" + "="*70)
    print("SECTION 4: LOG SUMMARIZATION")
    print("="*70)
    
    llm = BedrockLLM()
    summarizer = LogSummarizer(llm)
    
    # Get various log subsets for summarization
    error_logs = log_store.get_by_level("ERROR") + log_store.get_by_level("CRITICAL")
    db_related = [log for log in log_store.logs if "db" in log.service or "database" in log.message.lower()]
    
    # Timeline summary
    print("\n--- Timeline Summary ---")
    print("(Using first 10 logs)")
    timeline = summarizer.timeline_summary(log_store.logs[:10])
    print(f"\n{timeline}")
    
    # Error summary
    print("\n--- Error Summary ---")
    error_summary = summarizer.error_summary(error_logs[:10])
    print(f"\n{error_summary}")
    
    # Impact summary
    print("\n--- Impact Summary ---")
    impact = summarizer.impact_summary(error_logs[:8])
    print(f"\n{impact}")
    
    # Root cause summary
    print("\n--- Root Cause Analysis ---")
    sample_metrics = {
        "cpu_percent": 45,
        "memory_percent": 92,
        "db_connection_pool_used": 50,
        "db_connection_pool_max": 50,
        "request_queue_depth": 234
    }
    root_cause = summarizer.root_cause_summary(db_related[:6], sample_metrics)
    print(f"\n{root_cause}")
    
    return summarizer


# =============================================================================
# SECTION 5: SEMANTIC ERROR CLASSIFICATION
# =============================================================================

class SemanticErrorClassifier:
    """
    Classify errors by type, severity, and likely cause.
    
    Uses embeddings for similarity-based classification and
    LLM for more nuanced analysis when needed.
    """
    
    # Predefined error categories with example descriptions
    ERROR_CATEGORIES = {
        "authentication": [
            "login failed invalid password",
            "authentication rejected token expired",
            "access denied permission required",
            "unauthorized request missing token",
            "OAuth authentication failure"
        ],
        "database": [
            "database connection refused",
            "query timeout exceeded",
            "connection pool exhausted",
            "deadlock detected transaction",
            "SQL error database unavailable"
        ],
        "memory": [
            "out of memory heap space",
            "memory allocation failed",
            "garbage collection overhead",
            "memory limit exceeded OOM"
        ],
        "network": [
            "connection timeout service unreachable",
            "socket timeout no response",
            "DNS resolution failed",
            "HTTP 503 service unavailable",
            "high latency slow response"
        ],
        "application": [
            "null pointer exception",
            "validation error invalid input",
            "illegal state invalid transition",
            "division by zero arithmetic error"
        ]
    }
    
    SEVERITY_INDICATORS = {
        "critical": ["OutOfMemoryError", "CRITICAL", "fatal", "crash", "data loss", "corruption"],
        "high": ["ERROR", "failed", "exception", "timeout", "refused"],
        "medium": ["WARN", "warning", "retry", "degraded", "slow"],
        "low": ["INFO", "notice", "recovered", "retry succeeded"]
    }
    
    def __init__(self, embedder: BedrockEmbeddings, llm: BedrockLLM = None):
        self.embedder = embedder
        self.llm = llm
        self.category_embeddings = {}
        self._build_category_embeddings()
    
    def _build_category_embeddings(self):
        """Pre-compute embeddings for category examples."""
        print("   Building category embeddings...")
        for category, examples in self.ERROR_CATEGORIES.items():
            embeddings = self.embedder.embed_batch(examples)
            # Use mean embedding as category centroid
            self.category_embeddings[category] = np.mean(embeddings, axis=0)
    
    def classify_type(self, log: LogEntry) -> Tuple[str, float]:
        """
        Classify error type using embedding similarity.
        
        Returns (category, confidence_score).
        """
        best_category = "unknown"
        best_score = 0.0
        
        for category, centroid in self.category_embeddings.items():
            similarity = cosine_similarity(
                log.embedding.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_category = category
        
        # Threshold for unknown classification
        if best_score < 0.3:
            best_category = "unknown"
        
        return best_category, best_score
    
    def classify_severity(self, log: LogEntry) -> str:
        """
        Classify severity based on keywords and log level.
        """
        message_lower = log.message.lower()
        
        for severity, indicators in self.SEVERITY_INDICATORS.items():
            for indicator in indicators:
                if indicator.lower() in message_lower or indicator == log.level:
                    return severity
        
        return "medium"  # Default severity
    
    def classify_with_llm(self, log: LogEntry) -> Dict:
        """
        Get detailed classification using LLM.
        
        More accurate but slower than embedding-based classification.
        """
        if not self.llm:
            raise ValueError("LLM not configured")
        
        system_prompt = """You are an expert at classifying system errors.
        Analyze the error and provide structured classification."""
        
        prompt = f"""Classify this log entry:

[{log.level}] {log.service}: {log.message}

Return JSON with:
- error_type: category (authentication, database, memory, network, application, other)
- severity: critical/high/medium/low
- likely_cause: brief description of probable cause
- recommended_action: what to do next
- is_actionable: true if immediate action needed

Return only valid JSON."""

        response = self.llm.invoke(prompt, system_prompt, temperature=0.1)
        
        try:
            response = response.strip()
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse", "raw": response}
    
    def detect_novelty(
        self,
        log: LogEntry,
        known_logs: List[LogEntry],
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Detect if a log represents a novel (previously unseen) error type.
        
        Novel errors are potentially more important - they may represent
        new failure modes or emerging problems.
        """
        if not known_logs:
            return True, 1.0
        
        # Find maximum similarity to known logs
        max_similarity = 0.0
        for known in known_logs:
            similarity = cosine_similarity(
                log.embedding.reshape(1, -1),
                known.embedding.reshape(1, -1)
            )[0][0]
            max_similarity = max(max_similarity, similarity)
        
        # If max similarity is below threshold, it's novel
        is_novel = max_similarity < threshold
        novelty_score = 1.0 - max_similarity
        
        return is_novel, novelty_score


def demo_error_classification(log_store: SemanticLogStore):
    """Demonstrate error classification."""
    print("\n" + "="*70)
    print("SECTION 5: SEMANTIC ERROR CLASSIFICATION")
    print("="*70)
    
    embedder = BedrockEmbeddings()
    llm = BedrockLLM()
    classifier = SemanticErrorClassifier(embedder, llm)
    
    # Get error logs
    error_logs = [log for log in log_store.logs if log.level in ["ERROR", "CRITICAL", "WARN"]]
    
    print("\n--- Embedding-Based Classification ---")
    for log in error_logs[:8]:
        category, confidence = classifier.classify_type(log)
        severity = classifier.classify_severity(log)
        print(f"\n📝 {log.message[:60]}...")
        print(f"   Type: {category} (confidence: {confidence:.3f})")
        print(f"   Severity: {severity}")
    
    print("\n--- LLM-Enhanced Classification ---")
    sample_log = error_logs[0]
    print(f"\n📝 {sample_log.message}")
    classification = classifier.classify_with_llm(sample_log)
    print(f"   Classification: {json.dumps(classification, indent=6)}")
    
    print("\n--- Novelty Detection ---")
    # Use first 5 error logs as "known", test against the rest
    known_logs = error_logs[:5]
    test_logs = error_logs[5:10]
    
    print(f"Testing {len(test_logs)} logs against {len(known_logs)} known patterns:")
    for log in test_logs:
        is_novel, novelty_score = classifier.detect_novelty(log, known_logs)
        status = "🆕 NOVEL" if is_novel else "✓ Known pattern"
        print(f"   {status} (score: {novelty_score:.3f}): {log.message[:50]}...")
    
    return classifier


# =============================================================================
# SECTION 6: PUTTING IT ALL TOGETHER
# =============================================================================

class SemanticLogAnalyzer:
    """
    Complete semantic log analysis system.
    
    Combines all capabilities:
    - Semantic search
    - Entity extraction
    - Natural language queries
    - Summarization
    - Classification
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        print("Initializing Semantic Log Analyzer...")
        self.embedder = BedrockEmbeddings(region_name)
        self.llm = BedrockLLM(region_name)
        self.log_store = SemanticLogStore(self.embedder)
        self.entity_extractor = EntityExtractor(self.llm)
        self.nl_query = None  # Initialized after logs are loaded
        self.summarizer = LogSummarizer(self.llm)
        self.classifier = None  # Initialized after logs are loaded
    
    def ingest_logs(self, logs: List[Dict]):
        """Ingest logs and initialize all components."""
        print(f"Ingesting {len(logs)} logs...")
        self.log_store.ingest_batch(logs)
        
        print("Initializing query interface...")
        self.nl_query = NaturalLanguageLogQuery(self.llm, self.log_store)
        
        print("Initializing classifier...")
        self.classifier = SemanticErrorClassifier(self.embedder, self.llm)
        
        print("Ready!")
    
    def analyze_incident(self, query: str) -> Dict:
        """
        Comprehensive incident analysis from a natural language query.
        
        Returns:
        - Relevant logs
        - Extracted entities
        - Error classifications
        - Summary and root cause hypothesis
        """
        print(f"\n🔍 Analyzing: \"{query}\"")
        
        # Find relevant logs
        search_result = self.nl_query.execute_query(query, top_k=15)
        relevant_logs = [log for log, _ in search_result["results"]]
        
        if not relevant_logs:
            return {"error": "No relevant logs found"}
        
        # Extract entities from all logs
        all_entities = defaultdict(set)
        for log in relevant_logs:
            entities = self.entity_extractor.extract_with_regex(log.message)
            for host in entities.hosts:
                all_entities["hosts"].add(host)
            for user in entities.users:
                all_entities["users"].add(user)
            for ip in entities.ips:
                all_entities["ips"].add(ip)
        
        # Classify errors
        classifications = []
        for log in relevant_logs:
            if log.level in ["ERROR", "CRITICAL"]:
                category, confidence = self.classifier.classify_type(log)
                classifications.append({
                    "message": log.message[:80],
                    "category": category,
                    "confidence": confidence
                })
        
        # Generate summary
        summary = self.summarizer.root_cause_summary(relevant_logs)
        
        return {
            "query": query,
            "interpretation": search_result["interpretation"],
            "log_count": len(relevant_logs),
            "entities": {k: list(v) for k, v in all_entities.items()},
            "error_classifications": classifications,
            "root_cause_analysis": summary
        }


def demo_integrated_analysis():
    """Demonstrate the complete integrated system."""
    print("\n" + "="*70)
    print("SECTION 6: INTEGRATED SEMANTIC LOG ANALYSIS")
    print("="*70)
    
    # Initialize the complete system
    analyzer = SemanticLogAnalyzer()
    analyzer.ingest_logs(SAMPLE_LOGS)
    
    # Run comprehensive analysis
    analysis = analyzer.analyze_incident("database connectivity problems causing service failures")
    
    print("\n" + "="*70)
    print("📊 ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n🔍 Query: {analysis['query']}")
    print(f"📋 Interpretation: {analysis['interpretation'].get('semantic_query', 'N/A')}")
    print(f"📊 Logs found: {analysis['log_count']}")
    
    if analysis.get("entities"):
        print("\n🏷️  Extracted Entities:")
        for entity_type, values in analysis["entities"].items():
            if values:
                print(f"   {entity_type}: {values}")
    
    if analysis.get("error_classifications"):
        print("\n⚠️  Error Classifications:")
        for c in analysis["error_classifications"][:5]:
            print(f"   [{c['category']}] {c['message'][:50]}...")
    
    print("\n🔬 Root Cause Analysis:")
    print(analysis.get("root_cause_analysis", "N/A"))
    
    return analyzer


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run all demonstrations.
    
    Each section builds on previous ones:
    1. Semantic Search - Foundation of meaning-based log analysis
    2. Entity Extraction - Identify important elements in logs
    3. Natural Language Query - Ask questions in plain English
    4. Summarization - Condense logs into narratives
    5. Classification - Categorize errors automatically
    6. Integration - All capabilities working together
    """
    print("="*70)
    print("CHAPTER 4: SEMANTIC LOG UNDERSTANDING")
    print("Code Examples with Amazon Bedrock")
    print("="*70)
    
    try:
        # Section 1: Semantic Search
        log_store = demo_semantic_search()
        
        # Section 2: Entity Extraction
        demo_entity_extraction()
        
        # Section 3: Natural Language Querying
        demo_natural_language_query(log_store)
        
        # Section 4: Log Summarization
        demo_log_summarization(log_store)
        
        # Section 5: Error Classification
        demo_error_classification(log_store)
        
        # Section 6: Integrated System
        demo_integrated_analysis()
        
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
