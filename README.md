# AI-Powered Observability: Code Examples

Companion code for the book **AI-Powered Observability: From Noise to Insight** by Drew Moisant.

## About

This repository contains production-ready Python code examples demonstrating AI and machine learning techniques for observability, monitoring, and security operations. Each file corresponds to a chapter of the book and includes complete, runnable implementations.

## Chapters

| File | Chapter | Topics |
|------|---------|--------|
| `chapter_03_building_blocks.py` | Ch 3: Building Blocks | Embeddings, vector search, LLMs, anomaly detection, clustering |
| `chapter_04_semantic_log_understanding.py` | Ch 4: Semantic Log Understanding | Semantic search, log classification, entity extraction |
| `chapter_05_pattern_recognition.py` | Ch 5: Pattern Recognition at Scale | Drain parser, log clustering, cluster evolution |
| `chapter_06_log_analysis_pipeline.py` | Ch 6: Log Analysis Pipeline | End-to-end pipeline: parse, enrich, embed, classify |
| `chapter_07_statistical_vs_ml.py` | Ch 7: Statistical vs ML Approaches | Z-score, IQR, Isolation Forest, LOF, tiered detection |
| `chapter_08_time_series_anomaly_detection.py` | Ch 8: Time Series Anomaly Detection | Moving average, STL decomposition, LSTM autoencoder |
| `chapter_09_behavioral_anomaly_detection.py` | Ch 9: Behavioral Anomaly Detection | User profiling, session analysis, graph topology anomalies |
| `chapter_10_alert_fatigue_problem.py` | Ch 10: Alert Fatigue | Fatigue scoring, alert quality metrics, noise analysis |
| `chapter_11_correlation_deduplication.py` | Ch 11: Correlation & Deduplication | Deduplication, multi-signal correlation graphs |
| `chapter_12_predictive_alerting.py` | Ch 12: Predictive Alerting | Resource exhaustion prediction, pre-failure classification |
| `chapter_13_ai_performance_analysis.py` | Ch 13: Performance Analysis | Trace analysis, N+1 detection, regression detection |
| `chapter_14_capacity_planning.py` | Ch 14: Capacity Planning | Demand forecasting, scaling recommendations |
| `chapter_15_automated_remediation.py` | Ch 15: Automated Remediation | Circuit breaker, pattern-based action selection |
| `chapter_16_threat_detection.py` | Ch 16: Threat Detection | Beaconing detection, kill chain correlation, endpoint threats |
| `chapter_17_security_log_analysis.py` | Ch 17: Security Log Analysis | Template extraction, entity correlation, threat hunting |
| `chapter_18_ueba.py` | Ch 18: UEBA | Impossible travel, behavioral baselines, risk scoring |
| `chapter_19_incident_response.py` | Ch 19: Incident Response | Safety controllers, playbook execution, SOAR patterns |

## Prerequisites

### Python

Python 3.10 or later.

### Dependencies

```bash
pip install boto3 numpy pandas scikit-learn scipy statsmodels networkx
```

Optional (for LSTM-based detection in Chapter 8):
```bash
pip install tensorflow
```

### AWS Configuration

These examples use **Amazon Bedrock** for embeddings and LLM capabilities. You need:

1. An AWS account with Bedrock access enabled
2. AWS credentials configured (e.g., via `aws configure` or environment variables)
3. The following models enabled in your Bedrock console:
   - `amazon.titan-embed-text-v2:0` (embeddings)
   - `anthropic.claude-sonnet-4-20250514-v1:0` (LLM tasks)

Code that does not call Bedrock (anomaly detection algorithms, statistical methods, pattern recognition) will work without AWS credentials.

## Running Tests

```bash
python test_all_chapters.py
```

Tests exercise all core algorithms without making LLM API calls (Bedrock methods are mocked where needed). All 30 tests should pass.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

These code examples use or are informed by the following:

### Algorithms and Papers

- **Drain log parser**: He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). "Drain: An Online Log Parsing Approach with Fixed Depth Tree." IEEE ICWS. Canonical implementation: [LogPAI/logparser](https://github.com/logpai/logparser).
- **Isolation Forest**: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). "Isolation Forest." IEEE ICDM.
- **LSTM Autoencoder**: Architecture follows the Keras time series anomaly detection examples. See also Malhotra, P., et al. (2016). "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection."
- **Fourier seasonality features**: Hyndman, R. J., & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice." 3rd ed., OTexts.
- **Critical path analysis**: Informed by Google's Dapper paper: Sigelman, B. H., et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure."

### Security Frameworks

- **MITRE ATT&CK**: Attack stages and technique identifiers (e.g., T1048, T1071, T1078) are from the [MITRE ATT&CK](https://attack.mitre.org/) knowledge base. MITRE ATT&CK is a registered trademark of The MITRE Corporation.
- **Cyber Kill Chain**: Kill chain detection concept derives from Hutchins, E. M., Cloppert, M. J., & Amin, R. M. (2011). "Intelligence-Driven Computer Network Defense." Lockheed Martin Corporation.
- **Incident Response**: Response lifecycle follows NIST SP 800-61 Rev. 2, "Computer Security Incident Handling Guide."
- **Detection rules**: Endpoint detection patterns are informed by [Sigma rules](https://github.com/SigmaHQ/sigma) and Red Canary Threat Detection Reports.
- **Beaconing detection**: Interval analysis approach informed by [RITA](https://github.com/activecm/rita) (Real Intelligence Threat Analytics) and SANS Institute research.

### Libraries

Built on open source libraries including [scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/), [statsmodels](https://www.statsmodels.org/), [NetworkX](https://networkx.org/), and [TensorFlow/Keras](https://www.tensorflow.org/).

## Book

Get the book: *AI-Powered Observability: From Noise to Insight* — available on Amazon and Leanpub.
