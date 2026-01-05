"""
Chapter 6: Building a Log Analysis Pipeline
============================================

Code examples demonstrating a production-ready log analysis pipeline:
- Collection layer with context preservation
- Processing layer (parsing, enrichment, embedding, classification)
- Storage layer (raw logs, vector database, metadata)
- Analysis layer (real-time and batch)
- Query and interaction layer
- Feedback loop for continuous improvement
- Pipeline monitoring and observability

Prerequisites:
    pip install boto3 numpy scikit-learn

AWS Configuration:
    Ensure you have AWS credentials configured with access to Amazon Bedrock.
    Enable the following models in your Bedrock console:
    - amazon.titan-embed-text-v2:0 (for embeddings)
    - anthropic.claude-3-sonnet-20240229-v1:0 (for LLM tasks)

This chapter integrates concepts from Chapters 3-5 into a complete system.
"""

import json
import re
import time
import hashlib
import threading
import queue
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import numpy as np
import boto3
from sklearn.metrics.pairwise import cosine_similarity


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
# SECTION 1: DATA MODELS
# =============================================================================

class LogLevel(Enum):
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RawLog:
    """Raw log as received from collection."""
    raw_text: str
    source: str
    collected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedLog:
    """Log after parsing stage."""
    id: str
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    raw_text: str
    source: str
    collected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Parsing artifacts
    is_multiline: bool = False
    json_payload: Optional[Dict] = None


@dataclass
class EnrichedLog:
    """Log after enrichment stage."""
    # Inherit all parsed fields
    id: str
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    raw_text: str
    source: str
    collected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enrichment artifacts
    environment: str = "unknown"
    region: str = "unknown"
    version: str = "unknown"
    pod_id: str = None
    trace_id: str = None
    request_id: str = None
    
    # Extracted entities
    entities: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ProcessedLog:
    """Fully processed log ready for storage and analysis."""
    # Core fields
    id: str
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    raw_text: str
    
    # Context
    source: str
    collected_at: datetime
    processed_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enrichment
    environment: str = "unknown"
    region: str = "unknown"
    version: str = "unknown"
    trace_id: str = None
    request_id: str = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    
    # AI processing artifacts
    embedding: np.ndarray = None
    template_id: str = None
    template: str = None
    classification: Dict[str, Any] = field(default_factory=dict)
    anomaly_score: float = 0.0


# =============================================================================
# SECTION 2: COLLECTION LAYER
# =============================================================================

class LogCollector(ABC):
    """Abstract base class for log collectors."""
    
    @abstractmethod
    def collect(self) -> List[RawLog]:
        """Collect logs from source."""
        pass


class FileLogCollector(LogCollector):
    """
    Collect logs from files.
    
    Preserves:
    - Multiline integrity (stack traces)
    - Original timestamps
    - Source metadata
    """
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.multiline_patterns = [
            r'^\s+at\s+',  # Java stack traces
            r'^\s+File\s+',  # Python stack traces
            r'^\s+\.\.\.',  # Continuation lines
            r'^\t',  # Tab-indented lines
        ]
    
    def parse_raw_lines(self, lines: List[str]) -> List[RawLog]:
        """Parse lines, preserving multiline records."""
        raw_logs = []
        current_log = []
        
        for line in lines:
            line = line.rstrip('\n\r')
            
            # Check if this is a continuation line
            is_continuation = any(
                re.match(pattern, line) 
                for pattern in self.multiline_patterns
            )
            
            if is_continuation and current_log:
                # Append to current multiline log
                current_log.append(line)
            else:
                # Start a new log entry
                if current_log:
                    raw_logs.append(RawLog(
                        raw_text='\n'.join(current_log),
                        source=self.source_name,
                        collected_at=datetime.now(),
                        metadata={'multiline': len(current_log) > 1}
                    ))
                current_log = [line] if line.strip() else []
        
        # Don't forget the last log
        if current_log:
            raw_logs.append(RawLog(
                raw_text='\n'.join(current_log),
                source=self.source_name,
                collected_at=datetime.now(),
                metadata={'multiline': len(current_log) > 1}
            ))
        
        return raw_logs
    
    def collect(self) -> List[RawLog]:
        """Collect from simulated file source."""
        # In production, this would read from actual files
        # For demo, we return simulated logs
        return []


class KafkaSimulator:
    """
    Simulates Kafka-like message queue behavior.
    
    In production, replace with actual Kafka client:
    - confluent_kafka
    - kafka-python
    """
    
    def __init__(self, topic: str):
        self.topic = topic
        self.queue = queue.Queue()
        self.offset = 0
    
    def produce(self, message: Dict):
        """Add message to queue."""
        self.queue.put({
            'topic': self.topic,
            'offset': self.offset,
            'timestamp': datetime.now().isoformat(),
            'value': message
        })
        self.offset += 1
    
    def consume(self, timeout: float = 1.0) -> Optional[Dict]:
        """Consume message from queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def consume_batch(self, max_messages: int = 100, timeout: float = 1.0) -> List[Dict]:
        """Consume batch of messages."""
        messages = []
        deadline = time.time() + timeout
        
        while len(messages) < max_messages and time.time() < deadline:
            msg = self.consume(timeout=0.1)
            if msg:
                messages.append(msg)
            elif not messages:
                continue
            else:
                break
        
        return messages


class IngestionBuffer:
    """
    Buffer for log ingestion with backpressure handling.
    
    Provides resilience against processing slowdowns.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.dropped_count = 0
        self.total_ingested = 0
    
    def ingest(self, log: RawLog) -> bool:
        """
        Ingest a log into the buffer.
        
        Returns True if accepted, False if dropped due to backpressure.
        """
        try:
            self.buffer.put_nowait(log)
            self.total_ingested += 1
            return True
        except queue.Full:
            self.dropped_count += 1
            return False
    
    def get_batch(self, batch_size: int = 100, timeout: float = 1.0) -> List[RawLog]:
        """Get a batch of logs for processing."""
        batch = []
        deadline = time.time() + timeout
        
        while len(batch) < batch_size and time.time() < deadline:
            try:
                log = self.buffer.get(timeout=0.1)
                batch.append(log)
            except queue.Empty:
                if batch:
                    break
        
        return batch
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            'current_size': self.buffer.qsize(),
            'max_size': self.max_size,
            'total_ingested': self.total_ingested,
            'dropped_count': self.dropped_count,
            'drop_rate': self.dropped_count / max(1, self.total_ingested)
        }


# =============================================================================
# SECTION 3: PROCESSING LAYER
# =============================================================================

class LogParser:
    """
    Parse raw logs into structured format.
    
    Handles:
    - Multiple log formats
    - JSON payloads
    - Multiline logs (stack traces)
    - Timestamp parsing
    """
    
    # Common log format patterns
    PATTERNS = [
        # ISO timestamp with level and service
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+'
        r'(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\s+'
        r'(?P<service>[\w\-\.]+):\s*(?P<message>.*)$',
        
        # Level first, then timestamp
        r'^(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\s+'
        r'\[(?P<timestamp>[^\]]+)\]\s+'
        r'(?P<service>[\w\-\.]+)\s*-\s*(?P<message>.*)$',
        
        # Simple format: level message
        r'^(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\s+(?P<message>.*)$',
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]
        self.parse_count = 0
        self.failed_count = 0
    
    def parse(self, raw_log: RawLog) -> ParsedLog:
        """Parse a raw log into structured format."""
        self.parse_count += 1
        
        text = raw_log.raw_text
        first_line = text.split('\n')[0] if '\n' in text else text
        
        # Try each pattern
        for pattern in self.compiled_patterns:
            match = pattern.match(first_line)
            if match:
                groups = match.groupdict()
                
                # Parse timestamp
                timestamp = self._parse_timestamp(groups.get('timestamp'))
                
                # Parse level
                level = self._parse_level(groups.get('level', 'INFO'))
                
                # Extract JSON payload if present
                message = groups.get('message', text)
                json_payload = self._extract_json(message)
                
                return ParsedLog(
                    id=self._generate_id(raw_log),
                    timestamp=timestamp,
                    level=level,
                    service=groups.get('service', 'unknown'),
                    message=message,
                    raw_text=raw_log.raw_text,
                    source=raw_log.source,
                    collected_at=raw_log.collected_at,
                    metadata=raw_log.metadata,
                    is_multiline='\n' in text,
                    json_payload=json_payload
                )
        
        # No pattern matched - create basic parsed log
        self.failed_count += 1
        return ParsedLog(
            id=self._generate_id(raw_log),
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            service='unknown',
            message=text,
            raw_text=raw_log.raw_text,
            source=raw_log.source,
            collected_at=raw_log.collected_at,
            metadata={**raw_log.metadata, 'parse_failed': True},
            is_multiline='\n' in text
        )
    
    def _parse_timestamp(self, ts_str: Optional[str]) -> datetime:
        """Parse timestamp string to datetime."""
        if not ts_str:
            return datetime.now()
        
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(ts_str.replace('+00:00', 'Z').replace('Z', '+0000'), fmt.replace('Z', '%z'))
            except ValueError:
                continue
        
        return datetime.now()
    
    def _parse_level(self, level_str: str) -> LogLevel:
        """Parse log level string to enum."""
        level_map = {
            'DEBUG': LogLevel.DEBUG,
            'INFO': LogLevel.INFO,
            'WARN': LogLevel.WARN,
            'WARNING': LogLevel.WARN,
            'ERROR': LogLevel.ERROR,
            'CRITICAL': LogLevel.CRITICAL,
            'FATAL': LogLevel.CRITICAL,
        }
        return level_map.get(level_str.upper(), LogLevel.INFO)
    
    def _extract_json(self, message: str) -> Optional[Dict]:
        """Extract JSON payload from message if present."""
        # Look for JSON object in message
        json_match = re.search(r'\{[^{}]*\}', message)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
    
    def _generate_id(self, raw_log: RawLog) -> str:
        """Generate unique ID for log entry."""
        content = f"{raw_log.collected_at.isoformat()}{raw_log.raw_text}{raw_log.source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_stats(self) -> Dict:
        """Get parsing statistics."""
        return {
            'total_parsed': self.parse_count,
            'failed_count': self.failed_count,
            'success_rate': (self.parse_count - self.failed_count) / max(1, self.parse_count)
        }


class LogEnricher:
    """
    Enrich logs with additional context.
    
    Adds:
    - Environment/region/version from metadata
    - Extracted entities (IPs, users, request IDs)
    - Correlation IDs
    """
    
    def __init__(self):
        # Service to metadata mapping (in production, fetch from service registry)
        self.service_metadata = {
            'api-gateway': {'environment': 'production', 'region': 'us-east-1', 'version': '2.4.1'},
            'auth-service': {'environment': 'production', 'region': 'us-east-1', 'version': '1.8.0'},
            'order-service': {'environment': 'production', 'region': 'us-east-1', 'version': '3.2.0'},
            'payment-service': {'environment': 'production', 'region': 'us-east-1', 'version': '2.1.5'},
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'ip': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'request_id': r'\b(?:req|request)[_-]?([a-zA-Z0-9]{8,})\b',
            'trace_id': r'\b(?:trace|correlation)[_-]?id[=: ]+([a-zA-Z0-9\-]{16,})\b',
            'user_id': r'\buser[_-]?(?:id)?[=: ]+([a-zA-Z0-9\-]+)\b',
            'order_id': r'\b(?:order|ORD)[_-]?([A-Z0-9]{5,})\b',
        }
    
    def enrich(self, parsed_log: ParsedLog) -> EnrichedLog:
        """Enrich a parsed log with additional context."""
        # Get service metadata
        service_meta = self.service_metadata.get(
            parsed_log.service, 
            {'environment': 'unknown', 'region': 'unknown', 'version': 'unknown'}
        )
        
        # Extract entities
        entities = self._extract_entities(parsed_log.message)
        
        # Extract correlation IDs
        trace_id = self._extract_first(self.entity_patterns['trace_id'], parsed_log.message)
        request_id = self._extract_first(self.entity_patterns['request_id'], parsed_log.message)
        
        return EnrichedLog(
            id=parsed_log.id,
            timestamp=parsed_log.timestamp,
            level=parsed_log.level,
            service=parsed_log.service,
            message=parsed_log.message,
            raw_text=parsed_log.raw_text,
            source=parsed_log.source,
            collected_at=parsed_log.collected_at,
            metadata=parsed_log.metadata,
            environment=service_meta['environment'],
            region=service_meta['region'],
            version=service_meta['version'],
            trace_id=trace_id,
            request_id=request_id,
            entities=entities
        )
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract all entities from text."""
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Handle both direct matches and group matches
                if isinstance(matches[0], tuple):
                    matches = [m[0] for m in matches]
                entities[entity_type] = list(set(matches))
        return entities
    
    def _extract_first(self, pattern: str, text: str) -> Optional[str]:
        """Extract first match of pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        return None


class EmbeddingProcessor:
    """
    Generate embeddings for logs using Amazon Bedrock.
    
    Optimizations:
    - Batching for efficiency
    - Caching to avoid re-embedding duplicates
    """
    
    def __init__(self, region_name: str = "us-east-1", cache_size: int = 10000):
        self.client = get_bedrock_client(region_name)
        self.model_id = EMBEDDING_MODEL_ID
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text with caching."""
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Generate embedding
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
        
        # Cache management
        if len(self.cache) >= self.cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.cache.keys())[:self.cache_size // 10]
            for key in keys_to_remove:
                del self.cache[key]
        
        self.cache[cache_key] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple texts.
        
        Note: Titan doesn't support native batching, so we process sequentially
        but leverage caching. For true batching, consider SageMaker endpoints.
        """
        return [self.embed(text) for text in texts]
    
    def get_stats(self) -> Dict:
        """Get embedding statistics."""
        total = self.cache_hits + self.cache_misses
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, total)
        }


class LogClassifier:
    """
    Classify logs by type and severity.
    
    Uses embedding similarity to predefined category examples.
    """
    
    CATEGORIES = {
        'authentication': {
            'examples': [
                "login failed invalid password",
                "authentication rejected token expired",
                "access denied permission required"
            ],
            'severity_boost': 0.5
        },
        'database': {
            'examples': [
                "database connection refused",
                "query timeout exceeded",
                "connection pool exhausted"
            ],
            'severity_boost': 0.8
        },
        'network': {
            'examples': [
                "connection timeout service unreachable",
                "socket timeout no response",
                "DNS resolution failed"
            ],
            'severity_boost': 0.6
        },
        'memory': {
            'examples': [
                "out of memory heap space",
                "memory allocation failed",
                "garbage collection overhead"
            ],
            'severity_boost': 1.0
        },
        'application': {
            'examples': [
                "null pointer exception",
                "validation error invalid input",
                "illegal state invalid transition"
            ],
            'severity_boost': 0.4
        }
    }
    
    def __init__(self, embedder: EmbeddingProcessor):
        self.embedder = embedder
        self.category_centroids = {}
        self._build_centroids()
    
    def _build_centroids(self):
        """Pre-compute category centroids."""
        for category, config in self.CATEGORIES.items():
            embeddings = self.embedder.embed_batch(config['examples'])
            self.category_centroids[category] = {
                'centroid': np.mean(embeddings, axis=0),
                'severity_boost': config['severity_boost']
            }
    
    def classify(self, log_embedding: np.ndarray, log_level: LogLevel) -> Dict[str, Any]:
        """
        Classify a log based on its embedding.
        
        Returns category, confidence, and adjusted severity.
        """
        best_category = 'other'
        best_score = 0.0
        
        for category, config in self.category_centroids.items():
            similarity = cosine_similarity(
                log_embedding.reshape(1, -1),
                config['centroid'].reshape(1, -1)
            )[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_category = category
        
        # Calculate severity score
        base_severity = {
            LogLevel.DEBUG: 0.1,
            LogLevel.INFO: 0.2,
            LogLevel.WARN: 0.5,
            LogLevel.ERROR: 0.8,
            LogLevel.CRITICAL: 1.0
        }.get(log_level, 0.5)
        
        severity_boost = self.category_centroids.get(best_category, {}).get('severity_boost', 0.5)
        adjusted_severity = min(1.0, base_severity * (1 + severity_boost * best_score))
        
        return {
            'category': best_category if best_score > 0.3 else 'other',
            'confidence': best_score,
            'severity_score': adjusted_severity,
            'severity_label': self._severity_label(adjusted_severity)
        }
    
    def _severity_label(self, score: float) -> str:
        """Convert severity score to label."""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'


class TemplateExtractor:
    """
    Extract log templates using simple pattern matching.
    
    Simplified version of Drain algorithm for pipeline integration.
    """
    
    def __init__(self):
        self.templates = {}
        self.template_counts = defaultdict(int)
    
    def extract(self, message: str) -> Tuple[str, str]:
        """
        Extract template from message.
        
        Returns (template_id, template_string).
        """
        # Normalize variable parts
        normalized = message
        
        # Replace common variable patterns
        normalized = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', normalized)
        normalized = re.sub(r'\b[0-9a-fA-F]{8,}\b', '<HEX>', normalized)
        normalized = re.sub(r'\b\d+(?:\.\d+)?\s*(?:ms|s|MB|GB|%)\b', '<NUM>', normalized)
        normalized = re.sub(r'\b\d+\b', '<NUM>', normalized)
        normalized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', normalized)
        
        # Generate template ID
        template_id = hashlib.md5(normalized.encode()).hexdigest()[:8]
        
        # Store template
        if template_id not in self.templates:
            self.templates[template_id] = normalized
        
        self.template_counts[template_id] += 1
        
        return template_id, self.templates[template_id]
    
    def get_top_templates(self, n: int = 10) -> List[Dict]:
        """Get most frequent templates."""
        sorted_templates = sorted(
            self.template_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                'template_id': tid,
                'template': self.templates[tid],
                'count': count
            }
            for tid, count in sorted_templates[:n]
        ]


class ProcessingPipeline:
    """
    Complete log processing pipeline.
    
    Orchestrates: parsing -> enrichment -> embedding -> classification -> template extraction
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        print("Initializing processing pipeline...")
        self.parser = LogParser()
        self.enricher = LogEnricher()
        self.embedder = EmbeddingProcessor(region_name)
        self.classifier = LogClassifier(self.embedder)
        self.template_extractor = TemplateExtractor()
        
        self.processed_count = 0
        self.processing_times = []
    
    def process(self, raw_log: RawLog) -> ProcessedLog:
        """Process a single raw log through all stages."""
        start_time = time.time()
        
        # Stage 1: Parse
        parsed = self.parser.parse(raw_log)
        
        # Stage 2: Enrich
        enriched = self.enricher.enrich(parsed)
        
        # Stage 3: Embed
        embed_text = f"{enriched.level.value} {enriched.service}: {enriched.message}"
        embedding = self.embedder.embed(embed_text)
        
        # Stage 4: Classify
        classification = self.classifier.classify(embedding, enriched.level)
        
        # Stage 5: Template extraction
        template_id, template = self.template_extractor.extract(enriched.message)
        
        # Create processed log
        processed = ProcessedLog(
            id=enriched.id,
            timestamp=enriched.timestamp,
            level=enriched.level,
            service=enriched.service,
            message=enriched.message,
            raw_text=enriched.raw_text,
            source=enriched.source,
            collected_at=enriched.collected_at,
            processed_at=datetime.now(),
            metadata=enriched.metadata,
            environment=enriched.environment,
            region=enriched.region,
            version=enriched.version,
            trace_id=enriched.trace_id,
            request_id=enriched.request_id,
            entities=enriched.entities,
            embedding=embedding,
            template_id=template_id,
            template=template,
            classification=classification,
            anomaly_score=0.0  # Calculated later in analysis layer
        )
        
        # Track metrics
        self.processed_count += 1
        self.processing_times.append(time.time() - start_time)
        
        return processed
    
    def process_batch(self, raw_logs: List[RawLog]) -> List[ProcessedLog]:
        """Process a batch of logs."""
        return [self.process(log) for log in raw_logs]
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'processed_count': self.processed_count,
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000 if self.processing_times else 0,
            'p99_processing_time_ms': np.percentile(self.processing_times, 99) * 1000 if self.processing_times else 0,
            'parser_stats': self.parser.get_stats(),
            'embedder_stats': self.embedder.get_stats(),
            'template_count': len(self.template_extractor.templates)
        }


# =============================================================================
# SECTION 4: STORAGE LAYER
# =============================================================================

class RawLogStore:
    """
    Store for raw log data.
    
    In production, use Elasticsearch, OpenSearch, or similar.
    """
    
    def __init__(self):
        self.logs: Dict[str, ProcessedLog] = {}
        self.indices = {
            'by_service': defaultdict(list),
            'by_level': defaultdict(list),
            'by_template': defaultdict(list),
            'by_timestamp': []  # Sorted list for time-range queries
        }
    
    def store(self, log: ProcessedLog):
        """Store a processed log."""
        self.logs[log.id] = log
        
        # Update indices
        self.indices['by_service'][log.service].append(log.id)
        self.indices['by_level'][log.level.value].append(log.id)
        self.indices['by_template'][log.template_id].append(log.id)
        self.indices['by_timestamp'].append((log.timestamp, log.id))
    
    def get(self, log_id: str) -> Optional[ProcessedLog]:
        """Get log by ID."""
        return self.logs.get(log_id)
    
    def search(
        self,
        service: str = None,
        level: str = None,
        template_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[ProcessedLog]:
        """Search logs with filters."""
        # Start with all IDs or filtered set
        if service:
            candidate_ids = set(self.indices['by_service'][service])
        elif level:
            candidate_ids = set(self.indices['by_level'][level])
        elif template_id:
            candidate_ids = set(self.indices['by_template'][template_id])
        else:
            candidate_ids = set(self.logs.keys())
        
        # Apply time filter
        if start_time or end_time:
            time_filtered = set()
            for ts, log_id in self.indices['by_timestamp']:
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                if log_id in candidate_ids:
                    time_filtered.add(log_id)
            candidate_ids = time_filtered
        
        # Return logs
        results = [self.logs[lid] for lid in list(candidate_ids)[:limit]]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def count(self) -> int:
        """Get total log count."""
        return len(self.logs)


class VectorStore:
    """
    Vector store for semantic search.
    
    In production, use Pinecone, Weaviate, OpenSearch with k-NN, etc.
    """
    
    def __init__(self):
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def store(self, log_id: str, embedding: np.ndarray, metadata: Dict):
        """Store embedding with metadata."""
        self.vectors[log_id] = embedding
        self.metadata[log_id] = metadata
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_fn: Callable[[Dict], bool] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors.
        
        Returns list of (log_id, similarity, metadata) tuples.
        """
        results = []
        
        for log_id, embedding in self.vectors.items():
            # Apply filter
            if filter_fn and not filter_fn(self.metadata[log_id]):
                continue
            
            # Calculate similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            
            results.append((log_id, similarity, self.metadata[log_id]))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def count(self) -> int:
        """Get vector count."""
        return len(self.vectors)


class MetadataStore:
    """
    Store for aggregated metadata and statistics.
    
    Tracks:
    - Template statistics
    - Service health metrics
    - Error trends
    """
    
    def __init__(self):
        self.template_stats = defaultdict(lambda: {'count': 0, 'services': set(), 'last_seen': None})
        self.service_stats = defaultdict(lambda: {'total': 0, 'errors': 0, 'warnings': 0})
        self.hourly_counts = defaultdict(lambda: defaultdict(int))
    
    def update(self, log: ProcessedLog):
        """Update metadata from processed log."""
        # Template stats
        self.template_stats[log.template_id]['count'] += 1
        self.template_stats[log.template_id]['services'].add(log.service)
        self.template_stats[log.template_id]['last_seen'] = log.timestamp
        
        # Service stats
        self.service_stats[log.service]['total'] += 1
        if log.level == LogLevel.ERROR:
            self.service_stats[log.service]['errors'] += 1
        elif log.level == LogLevel.WARN:
            self.service_stats[log.service]['warnings'] += 1
        
        # Hourly counts
        hour_bucket = log.timestamp.replace(minute=0, second=0, microsecond=0)
        self.hourly_counts[hour_bucket][log.service] += 1
    
    def get_template_stats(self) -> List[Dict]:
        """Get template statistics."""
        return [
            {
                'template_id': tid,
                'count': stats['count'],
                'services': list(stats['services']),
                'last_seen': stats['last_seen'].isoformat() if stats['last_seen'] else None
            }
            for tid, stats in sorted(
                self.template_stats.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
        ]
    
    def get_service_health(self) -> Dict[str, Dict]:
        """Get service health summary."""
        result = {}
        for service, stats in self.service_stats.items():
            error_rate = stats['errors'] / max(1, stats['total'])
            result[service] = {
                'total_logs': stats['total'],
                'error_count': stats['errors'],
                'warning_count': stats['warnings'],
                'error_rate': error_rate,
                'health': 'healthy' if error_rate < 0.01 else 'degraded' if error_rate < 0.05 else 'unhealthy'
            }
        return result


class StorageLayer:
    """
    Unified storage layer combining all stores.
    """
    
    def __init__(self):
        self.raw_store = RawLogStore()
        self.vector_store = VectorStore()
        self.metadata_store = MetadataStore()
    
    def store(self, log: ProcessedLog):
        """Store processed log in all stores."""
        # Raw store
        self.raw_store.store(log)
        
        # Vector store
        self.vector_store.store(
            log.id,
            log.embedding,
            {
                'service': log.service,
                'level': log.level.value,
                'template_id': log.template_id,
                'timestamp': log.timestamp.isoformat()
            }
        )
        
        # Metadata store
        self.metadata_store.update(log)
    
    def store_batch(self, logs: List[ProcessedLog]):
        """Store batch of logs."""
        for log in logs:
            self.store(log)
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            'raw_log_count': self.raw_store.count(),
            'vector_count': self.vector_store.count(),
            'template_count': len(self.metadata_store.template_stats),
            'service_count': len(self.metadata_store.service_stats)
        }


# =============================================================================
# SECTION 5: ANALYSIS LAYER
# =============================================================================

class RealTimeAnalyzer:
    """
    Real-time analysis of incoming logs.
    
    Performs:
    - Anomaly scoring
    - Alert triggering
    - Pattern matching
    """
    
    def __init__(self, storage: StorageLayer):
        self.storage = storage
        self.alert_thresholds = {
            'anomaly_score': 0.8,
            'error_rate_spike': 0.1,
            'new_template': True
        }
        self.known_templates = set()
        self.alerts = []
    
    def analyze(self, log: ProcessedLog) -> Dict:
        """
        Analyze a single log in real-time.
        
        Returns analysis results including any triggered alerts.
        """
        results = {
            'log_id': log.id,
            'anomaly_score': 0.0,
            'alerts': [],
            'insights': []
        }
        
        # Check for new template
        if log.template_id not in self.known_templates:
            self.known_templates.add(log.template_id)
            if self.alert_thresholds['new_template']:
                alert = {
                    'type': 'new_template',
                    'severity': 'info',
                    'message': f'New log template detected: {log.template[:50]}...',
                    'template_id': log.template_id
                }
                results['alerts'].append(alert)
                self.alerts.append(alert)
        
        # Calculate anomaly score based on classification
        classification = log.classification
        if classification.get('confidence', 0) < 0.3:
            # Low confidence = potentially anomalous
            results['anomaly_score'] = 0.7
            results['insights'].append('Log does not match known patterns well')
        
        if classification.get('severity_label') == 'critical':
            results['anomaly_score'] = max(results['anomaly_score'], 0.9)
            alert = {
                'type': 'critical_log',
                'severity': 'high',
                'message': f'Critical log detected in {log.service}',
                'log_id': log.id
            }
            results['alerts'].append(alert)
            self.alerts.append(alert)
        
        return results
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get recent alerts."""
        return self.alerts[-n:]


class BatchAnalyzer:
    """
    Batch analysis over stored logs.
    
    Performs:
    - Trend detection
    - Cluster analysis
    - Correlation discovery
    """
    
    def __init__(self, storage: StorageLayer):
        self.storage = storage
    
    def analyze_trends(self, hours: int = 24) -> Dict:
        """Analyze trends over recent hours."""
        # Get hourly counts
        hourly_data = self.storage.metadata_store.hourly_counts
        
        # Calculate trends per service
        service_trends = {}
        service_stats = self.storage.metadata_store.service_stats
        
        for service, stats in service_stats.items():
            error_rate = stats['errors'] / max(1, stats['total'])
            service_trends[service] = {
                'total_logs': stats['total'],
                'error_rate': error_rate,
                'trend': 'stable'  # Simplified - in production, calculate actual trend
            }
        
        return {
            'period_hours': hours,
            'service_trends': service_trends,
            'total_logs': sum(s['total'] for s in service_stats.values()),
            'total_errors': sum(s['errors'] for s in service_stats.values())
        }
    
    def find_correlations(self) -> List[Dict]:
        """Find correlated log patterns."""
        # Simplified correlation detection
        # In production, use more sophisticated methods
        
        template_stats = self.storage.metadata_store.template_stats
        correlations = []
        
        # Find templates that appear in multiple services
        for tid, stats in template_stats.items():
            if len(stats['services']) > 1:
                correlations.append({
                    'type': 'cross_service_pattern',
                    'template_id': tid,
                    'services': list(stats['services']),
                    'count': stats['count']
                })
        
        return correlations


# =============================================================================
# SECTION 6: QUERY AND INTERACTION LAYER
# =============================================================================

class QueryEngine:
    """
    Query interface for the log analysis system.
    
    Supports:
    - Keyword search
    - Semantic search
    - Natural language queries
    - Aggregations
    """
    
    def __init__(self, storage: StorageLayer, embedder: EmbeddingProcessor):
        self.storage = storage
        self.embedder = embedder
    
    def keyword_search(
        self,
        query: str,
        service: str = None,
        level: str = None,
        limit: int = 100
    ) -> List[ProcessedLog]:
        """Search logs by keyword."""
        # Get candidate logs
        logs = self.storage.raw_store.search(
            service=service,
            level=level,
            limit=limit * 2  # Get more than needed for filtering
        )
        
        # Filter by keyword
        query_lower = query.lower()
        results = [
            log for log in logs
            if query_lower in log.message.lower()
        ]
        
        return results[:limit]
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        service: str = None
    ) -> List[Tuple[ProcessedLog, float]]:
        """Search logs by semantic similarity."""
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Search vector store
        filter_fn = None
        if service:
            filter_fn = lambda meta: meta['service'] == service
        
        results = self.storage.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_fn=filter_fn
        )
        
        # Get full logs
        return [
            (self.storage.raw_store.get(log_id), score)
            for log_id, score, _ in results
            if self.storage.raw_store.get(log_id)
        ]
    
    def get_service_summary(self, service: str) -> Dict:
        """Get summary for a specific service."""
        health = self.storage.metadata_store.get_service_health()
        service_health = health.get(service, {})
        
        # Get recent logs
        recent_logs = self.storage.raw_store.search(service=service, limit=10)
        
        return {
            'service': service,
            'health': service_health,
            'recent_log_count': len(recent_logs),
            'recent_errors': [
                log.message[:100] for log in recent_logs
                if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
            ][:5]
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard display."""
        return {
            'storage_stats': self.storage.get_stats(),
            'service_health': self.storage.metadata_store.get_service_health(),
            'top_templates': self.storage.metadata_store.get_template_stats()[:10]
        }


# =============================================================================
# SECTION 7: FEEDBACK LOOP
# =============================================================================

@dataclass
class Feedback:
    """User feedback on log analysis."""
    log_id: str
    feedback_type: str  # 'classification', 'alert', 'search_relevance'
    is_correct: bool
    correct_value: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = "anonymous"


class FeedbackCollector:
    """
    Collect and process user feedback.
    
    Feedback is used to:
    - Improve classification models
    - Adjust alert thresholds
    - Refine search relevance
    """
    
    def __init__(self):
        self.feedback_log: List[Feedback] = []
        self.classification_corrections = defaultdict(list)
        self.alert_feedback = {'true_positive': 0, 'false_positive': 0}
    
    def record_feedback(self, feedback: Feedback):
        """Record user feedback."""
        self.feedback_log.append(feedback)
        
        if feedback.feedback_type == 'classification':
            self.classification_corrections[feedback.log_id].append({
                'is_correct': feedback.is_correct,
                'correct_value': feedback.correct_value
            })
        elif feedback.feedback_type == 'alert':
            if feedback.is_correct:
                self.alert_feedback['true_positive'] += 1
            else:
                self.alert_feedback['false_positive'] += 1
    
    def get_classification_accuracy(self) -> float:
        """Calculate classification accuracy from feedback."""
        if not self.classification_corrections:
            return 1.0
        
        correct = sum(
            1 for corrections in self.classification_corrections.values()
            for c in corrections if c['is_correct']
        )
        total = sum(len(c) for c in self.classification_corrections.values())
        
        return correct / max(1, total)
    
    def get_alert_precision(self) -> float:
        """Calculate alert precision from feedback."""
        tp = self.alert_feedback['true_positive']
        fp = self.alert_feedback['false_positive']
        return tp / max(1, tp + fp)
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics."""
        return {
            'total_feedback': len(self.feedback_log),
            'classification_accuracy': self.get_classification_accuracy(),
            'alert_precision': self.get_alert_precision(),
            'recent_feedback': len([
                f for f in self.feedback_log
                if f.timestamp > datetime.now() - timedelta(hours=24)
            ])
        }


# =============================================================================
# SECTION 8: PIPELINE ORCHESTRATOR
# =============================================================================

class LogAnalysisPipeline:
    """
    Complete log analysis pipeline orchestrator.
    
    Coordinates all components:
    - Collection
    - Processing
    - Storage
    - Analysis
    - Query
    - Feedback
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        print("="*60)
        print("Initializing Log Analysis Pipeline")
        print("="*60)
        
        # Initialize components
        print("\n📥 Setting up collection layer...")
        self.ingestion_buffer = IngestionBuffer(max_size=10000)
        
        print("⚙️  Setting up processing layer...")
        self.processing = ProcessingPipeline(region_name)
        
        print("💾 Setting up storage layer...")
        self.storage = StorageLayer()
        
        print("🔍 Setting up analysis layer...")
        self.realtime_analyzer = RealTimeAnalyzer(self.storage)
        self.batch_analyzer = BatchAnalyzer(self.storage)
        
        print("🔎 Setting up query layer...")
        self.query_engine = QueryEngine(self.storage, self.processing.embedder)
        
        print("📝 Setting up feedback loop...")
        self.feedback = FeedbackCollector()
        
        print("\n✅ Pipeline initialized!")
    
    def ingest(self, raw_text: str, source: str = "api", metadata: Dict = None) -> bool:
        """Ingest a raw log into the pipeline."""
        raw_log = RawLog(
            raw_text=raw_text,
            source=source,
            collected_at=datetime.now(),
            metadata=metadata or {}
        )
        return self.ingestion_buffer.ingest(raw_log)
    
    def process_pending(self, batch_size: int = 100) -> int:
        """Process pending logs from the ingestion buffer."""
        raw_logs = self.ingestion_buffer.get_batch(batch_size)
        
        if not raw_logs:
            return 0
        
        # Process
        processed_logs = self.processing.process_batch(raw_logs)
        
        # Store
        self.storage.store_batch(processed_logs)
        
        # Analyze
        for log in processed_logs:
            self.realtime_analyzer.analyze(log)
        
        return len(processed_logs)
    
    def search(self, query: str, method: str = "semantic", **kwargs) -> List:
        """Search logs."""
        if method == "semantic":
            return self.query_engine.semantic_search(query, **kwargs)
        else:
            return self.query_engine.keyword_search(query, **kwargs)
    
    def get_dashboard(self) -> Dict:
        """Get dashboard data."""
        return {
            **self.query_engine.get_dashboard_data(),
            'ingestion_stats': self.ingestion_buffer.get_stats(),
            'processing_stats': self.processing.get_stats(),
            'recent_alerts': self.realtime_analyzer.get_recent_alerts(),
            'feedback_stats': self.feedback.get_feedback_stats()
        }
    
    def record_feedback(self, log_id: str, feedback_type: str, is_correct: bool, correct_value: str = None):
        """Record user feedback."""
        self.feedback.record_feedback(Feedback(
            log_id=log_id,
            feedback_type=feedback_type,
            is_correct=is_correct,
            correct_value=correct_value
        ))


# =============================================================================
# SECTION 9: DEMO AND TESTING
# =============================================================================

def generate_sample_logs() -> List[str]:
    """Generate sample log entries for testing."""
    return [
        "2024-01-15T10:23:45.123Z ERROR auth-service: Login failed for user john@example.com - invalid password",
        "2024-01-15T10:23:46.456Z INFO api-gateway: Request completed: GET /api/users - 200 in 45ms",
        "2024-01-15T10:23:47.789Z ERROR order-service: Connection to postgres://db-primary:5432 failed: connection refused",
        "2024-01-15T10:23:48.012Z WARN api-gateway: Slow response: /api/search took 2500ms (threshold: 500ms)",
        "2024-01-15T10:23:49.345Z INFO auth-service: User alice@example.com logged in from 192.168.1.50",
        "2024-01-15T10:23:50.678Z ERROR payment-service: NullPointerException in PaymentProcessor.process() at line 234",
        "2024-01-15T10:23:51.901Z CRITICAL order-service: OutOfMemoryError: Java heap space - current heap 7.8GB/8GB",
        "2024-01-15T10:23:52.234Z INFO order-service: Order ORD-12345 processed successfully - total $149.99",
        "2024-01-15T10:23:53.567Z ERROR api-gateway: Connection timeout after 5000ms to payment-service:8080",
        "2024-01-15T10:23:54.890Z WARN order-service: Memory usage high: 7500MB / 8192MB (92%)",
        "2024-01-15T10:23:55.123Z DEBUG cache-service: Cache hit for key user:profile:12345 - response time 2ms",
        "2024-01-15T10:23:56.456Z ERROR auth-service: Authentication rejected: token expired for session sess_abc123",
    ]


def demo_pipeline():
    """Demonstrate the complete pipeline."""
    print("\n" + "="*70)
    print("LOG ANALYSIS PIPELINE DEMONSTRATION")
    print("="*70)
    
    # Initialize pipeline
    pipeline = LogAnalysisPipeline()
    
    # Generate and ingest sample logs
    print("\n" + "-"*50)
    print("STEP 1: Ingesting Sample Logs")
    print("-"*50)
    
    sample_logs = generate_sample_logs()
    for log_text in sample_logs:
        pipeline.ingest(log_text, source="demo")
    
    print(f"Ingested {len(sample_logs)} logs into buffer")
    
    # Process logs
    print("\n" + "-"*50)
    print("STEP 2: Processing Logs")
    print("-"*50)
    
    processed_count = pipeline.process_pending()
    print(f"Processed {processed_count} logs")
    
    # Show processing stats
    stats = pipeline.processing.get_stats()
    print(f"\nProcessing Statistics:")
    print(f"   Avg processing time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"   Parser success rate: {stats['parser_stats']['success_rate']*100:.1f}%")
    print(f"   Embedding cache hit rate: {stats['embedder_stats']['cache_hit_rate']*100:.1f}%")
    print(f"   Templates discovered: {stats['template_count']}")
    
    # Show storage stats
    print("\n" + "-"*50)
    print("STEP 3: Storage Status")
    print("-"*50)
    
    storage_stats = pipeline.storage.get_stats()
    print(f"Raw logs stored: {storage_stats['raw_log_count']}")
    print(f"Vectors stored: {storage_stats['vector_count']}")
    print(f"Unique templates: {storage_stats['template_count']}")
    print(f"Services tracked: {storage_stats['service_count']}")
    
    # Demonstrate semantic search
    print("\n" + "-"*50)
    print("STEP 4: Semantic Search")
    print("-"*50)
    
    queries = [
        "authentication failures",
        "database connection problems",
        "memory issues"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        results = pipeline.search(query, method="semantic", top_k=3)
        for log, score in results:
            print(f"   [{score:.3f}] [{log.level.value}] {log.message[:50]}...")
    
    # Show real-time analysis results
    print("\n" + "-"*50)
    print("STEP 5: Real-Time Analysis")
    print("-"*50)
    
    alerts = pipeline.realtime_analyzer.get_recent_alerts()
    print(f"\nRecent Alerts ({len(alerts)}):")
    for alert in alerts[:5]:
        print(f"   [{alert['severity'].upper()}] {alert['type']}: {alert['message'][:50]}...")
    
    # Show service health
    print("\n" + "-"*50)
    print("STEP 6: Service Health")
    print("-"*50)
    
    health = pipeline.storage.metadata_store.get_service_health()
    for service, status in health.items():
        icon = "🟢" if status['health'] == 'healthy' else "🟡" if status['health'] == 'degraded' else "🔴"
        print(f"   {icon} {service}: {status['health']} (error rate: {status['error_rate']*100:.1f}%)")
    
    # Show dashboard data
    print("\n" + "-"*50)
    print("STEP 7: Dashboard Summary")
    print("-"*50)
    
    dashboard = pipeline.get_dashboard()
    print("\n📊 Pipeline Dashboard:")
    print(f"   Total logs: {dashboard['storage_stats']['raw_log_count']}")
    print(f"   Templates: {dashboard['storage_stats']['template_count']}")
    print(f"   Alerts: {len(dashboard['recent_alerts'])}")
    print(f"   Services: {dashboard['storage_stats']['service_count']}")
    
    # Demonstrate feedback loop
    print("\n" + "-"*50)
    print("STEP 8: Feedback Loop")
    print("-"*50)
    
    # Simulate user feedback
    pipeline.record_feedback("log_001", "classification", True)
    pipeline.record_feedback("log_002", "alert", False)  # False positive
    pipeline.record_feedback("log_003", "alert", True)   # True positive
    
    feedback_stats = pipeline.feedback.get_feedback_stats()
    print(f"\nFeedback Statistics:")
    print(f"   Total feedback: {feedback_stats['total_feedback']}")
    print(f"   Classification accuracy: {feedback_stats['classification_accuracy']*100:.1f}%")
    print(f"   Alert precision: {feedback_stats['alert_precision']*100:.1f}%")
    
    return pipeline


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete pipeline demonstration."""
    print("="*70)
    print("CHAPTER 6: BUILDING A LOG ANALYSIS PIPELINE")
    print("Complete Production-Ready Implementation")
    print("="*70)
    
    try:
        pipeline = demo_pipeline()
        
        print("\n" + "="*70)
        print("✅ Pipeline demonstration completed successfully!")
        print("="*70)
        
        print("\n📋 Implementation Phases Demonstrated:")
        print("   Phase 1: ✅ Foundation (collection, parsing, storage)")
        print("   Phase 2: ✅ Embedding and Semantic Search")
        print("   Phase 3: ✅ Classification and Pattern Detection")
        print("   Phase 4: ⏳ Natural Language (requires LLM integration)")
        print("   Phase 5: ✅ Feedback and Continuous Improvement")
        print("   Phase 6: ⏳ Advanced Capabilities (predictive, remediation)")
        
        return pipeline
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure AWS credentials are configured")
        print("2. Enable Bedrock models in AWS Console")
        print("3. Check region supports Bedrock")
        raise


if __name__ == "__main__":
    main()
