"""
Loads synthetic runbooks for common failure categories.

Runbooks are stored as structured JSON in data/processed/runbooks/.
They supplement post-mortems with prescriptive remediation playbooks,
giving the retrieval system access to actionable steps even for failure
modes not well-covered by historical post-mortems.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNBOOK_DIR = ROOT / "data" / "processed" / "runbooks"

# Synthetic runbooks for the most common incident categories
RUNBOOKS: list[dict] = [
    {
        "id": "runbook_kafka_consumer_lag",
        "company": "internal",
        "failure_category": "messaging",
        "infrastructure_tags": ["Kafka", "consumer-group"],
        "title": "Kafka Consumer Lag Runbook",
        "services_affected": ["messaging", "event-processing"],
        "severity": "P1",
        "root_cause_summary": "Consumer lag accumulation due to slow processing, thread exhaustion, or broker-side issues.",
        "timeline": [],
        "remediation_steps": [
            "Check consumer group lag: `kafka-consumer-groups.sh --bootstrap-server ... --describe --group <group>`",
            "Identify slow partitions — uneven lag distribution points to data skew or hot partition",
            "Check consumer thread pool size vs. partition count — should be ≥ partitions",
            "Look for GC pauses in consumer JVM: `jstat -gcutil <pid> 1000`",
            "Check broker disk I/O and network saturation on lag-heavy partitions",
            "If processing is slow: scale out consumer instances or increase thread pool",
            "If broker-side: check ISR (in-sync replicas), replication lag, and disk usage",
            "Temporary relief: increase `max.poll.interval.ms` to prevent rebalances during catch-up",
        ],
        "error_codes": ["REBALANCE_IN_PROGRESS", "OFFSET_OUT_OF_RANGE"],
        "lessons_learned": [
            "Monitor consumer lag at partition level, not just group level",
            "Alert on lag velocity (rate of increase) not just absolute lag",
        ],
        "source": "synthetic_runbook",
        "raw_text": "",
    },
    {
        "id": "runbook_postgres_connection_exhaustion",
        "company": "internal",
        "failure_category": "database",
        "infrastructure_tags": ["PostgreSQL", "connection-pool", "pgbouncer"],
        "title": "PostgreSQL Connection Exhaustion Runbook",
        "services_affected": ["database", "api"],
        "severity": "P0",
        "root_cause_summary": "PostgreSQL max_connections reached; new connections rejected with 'too many clients' error.",
        "timeline": [],
        "remediation_steps": [
            "Check current connections: `SELECT count(*), state FROM pg_stat_activity GROUP BY state`",
            "Kill idle connections: `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < NOW() - INTERVAL '5 minutes'`",
            "Check PgBouncer pool usage: `SHOW POOLS` in pgbouncer console",
            "Verify `max_connections` setting vs. pool_size configuration",
            "Check for connection leaks: apps not returning connections to pool",
            "Temporarily increase `max_connections` if RDS/instance allows without restart (requires reboot on RDS)",
            "Scale out PgBouncer if it is the bottleneck",
            "Long-term: implement connection pooling at application layer (HikariCP, etc.)",
        ],
        "error_codes": ["ECONNREFUSED", "53300", "too many clients"],
        "lessons_learned": [
            "Alert at 80% connection utilization, not 100%",
            "PgBouncer transaction-mode pooling is more efficient than session-mode for microservices",
        ],
        "source": "synthetic_runbook",
        "raw_text": "",
    },
    {
        "id": "runbook_oom_killed_pod",
        "company": "internal",
        "failure_category": "compute",
        "infrastructure_tags": ["Kubernetes", "OOMKilled", "memory"],
        "title": "Kubernetes OOMKilled Pod Runbook",
        "services_affected": ["kubernetes", "compute"],
        "severity": "P1",
        "root_cause_summary": "Container memory limit exceeded; kubelet kills container with OOMKilled.",
        "timeline": [],
        "remediation_steps": [
            "Confirm OOMKill: `kubectl describe pod <pod> | grep -A5 OOMKilled`",
            "Check memory limit vs. request ratio — limits < 2x requests often cause spurious OOMs",
            "Profile memory usage: `kubectl exec <pod> -- cat /sys/fs/cgroup/memory/memory.usage_in_bytes`",
            "Check for memory leak: plot RSS over time in Prometheus (container_memory_rss)",
            "Increase memory limit as temporary relief (edit deployment)",
            "Check for JVM heap misconfiguration: MaxRAMPercentage should match container limit",
            "Enable Java's -XX:+ExitOnOutOfMemoryError to get heap dump instead of silent kill",
            "Long-term: identify and fix memory leak; right-size resource limits based on profiling",
        ],
        "error_codes": ["OOMKilled", "137"],
        "lessons_learned": [
            "Set resource.limits.memory 2–3x resource.requests.memory for bursty workloads",
            "Always configure JVM MaxRAMPercentage=75 in containerized apps",
        ],
        "source": "synthetic_runbook",
        "raw_text": "",
    },
    {
        "id": "runbook_redis_eviction",
        "company": "internal",
        "failure_category": "storage",
        "infrastructure_tags": ["Redis", "eviction", "cache"],
        "title": "Redis Cache Eviction Runbook",
        "services_affected": ["cache", "session"],
        "severity": "P1",
        "root_cause_summary": "Redis hitting maxmemory limit; keys being evicted, causing cache miss storm downstream.",
        "timeline": [],
        "remediation_steps": [
            "Check eviction rate: `redis-cli INFO stats | grep evicted_keys`",
            "Check memory fragmentation: `INFO memory | grep mem_fragmentation_ratio` — ratio > 1.5 indicates fragmentation",
            "Identify large keys: `redis-cli --bigkeys`",
            "Check TTL distribution — missing TTLs cause unbounded growth",
            "Increase maxmemory if headroom exists, or scale to Redis Cluster",
            "Review eviction policy: allkeys-lru is safer than noeviction for cache use cases",
            "Flush idle/cold keyspaces if segmented by prefix",
            "Consider read-through cache warming after eviction to prevent thundering herd",
        ],
        "error_codes": ["OOM command not allowed", "LOADING"],
        "lessons_learned": [
            "Always set TTLs on all cache keys",
            "Monitor memory fragmentation ratio, not just used_memory",
        ],
        "source": "synthetic_runbook",
        "raw_text": "",
    },
    {
        "id": "runbook_dns_resolution_failure",
        "company": "internal",
        "failure_category": "network",
        "infrastructure_tags": ["DNS", "CoreDNS", "network"],
        "title": "DNS Resolution Failure Runbook",
        "services_affected": ["dns", "networking"],
        "severity": "P0",
        "root_cause_summary": "DNS resolution failures causing NXDOMAIN or timeout errors across services.",
        "timeline": [],
        "remediation_steps": [
            "Verify DNS resolution: `nslookup <hostname> <dns-server>` from multiple pods",
            "Check CoreDNS pod health in k8s: `kubectl -n kube-system get pods -l k8s-app=kube-dns`",
            "Check CoreDNS logs: `kubectl -n kube-system logs -l k8s-app=kube-dns --tail=100`",
            "Verify ndots configuration — high ndots causes excessive search domain lookups",
            "Check upstream DNS (Route53, etc.) health and quotas",
            "Verify network policies aren't blocking UDP/53 to kube-dns",
            "Check DNS cache hit rate — cold start after pod restart causes spike",
            "Temporary: add explicit search domain or use FQDN (trailing dot) to bypass ndots",
        ],
        "error_codes": ["NXDOMAIN", "SERVFAIL", "ECONNREFUSED", "dial tcp: lookup"],
        "lessons_learned": [
            "Use ndots:2 in pod DNS config instead of default ndots:5",
            "Deploy redundant CoreDNS replicas with PodDisruptionBudget",
        ],
        "source": "synthetic_runbook",
        "raw_text": "",
    },
]


def load_runbooks(overwrite: bool = False) -> list[Path]:
    RUNBOOK_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for runbook in RUNBOOKS:
        out_path = RUNBOOK_DIR / f"{runbook['id']}.json"
        if not overwrite and out_path.exists():
            saved.append(out_path)
            continue
        runbook["raw_text"] = _build_raw_text(runbook)
        out_path.write_text(json.dumps(runbook, ensure_ascii=False, indent=2))
        saved.append(out_path)

    print(f"Loaded {len(saved)} runbooks to {RUNBOOK_DIR}")
    return saved


def _build_raw_text(r: dict) -> str:
    """Synthesize a readable text representation for embedding."""
    lines = [
        f"# {r['title']}",
        f"Category: {r['failure_category']}",
        f"Infrastructure: {', '.join(r['infrastructure_tags'])}",
        f"\n## Root Cause\n{r['root_cause_summary']}",
        "\n## Remediation Steps",
    ]
    for i, step in enumerate(r["remediation_steps"], 1):
        lines.append(f"{i}. {step}")
    if r.get("error_codes"):
        lines.append(f"\n## Error Codes\n{', '.join(r['error_codes'])}")
    if r.get("lessons_learned"):
        lines.append("\n## Lessons Learned")
        for lesson in r["lessons_learned"]:
            lines.append(f"- {lesson}")
    return "\n".join(lines)


if __name__ == "__main__":
    load_runbooks()
