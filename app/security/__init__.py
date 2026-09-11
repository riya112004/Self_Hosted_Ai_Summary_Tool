"""
Security layer: PII detection, sanitization, and role-based access.

Deployment note (see plan): the inference server (Ollama/vLLM) MUST be
isolated from outbound internet access - run it on an air-gapped / private
network host so raw data can never leak to external endpoints.
"""