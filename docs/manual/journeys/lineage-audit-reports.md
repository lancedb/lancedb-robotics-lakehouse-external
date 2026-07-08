# Journey: lineage audit reports and cleanup gates

**Scenario.** Before pruning old Lance table versions or cutting a release, you
need proof that lineage still resolves: graph edges point at real artifacts,
source coordinates are reachable, table-version pins can be read, external URNs
are not stale, and cleanup candidates are visible before deletion.

`lake.lineage.audit(...)` remains the quick report-only check. Recording the
report makes it durable in `lineage_audit_reports`, where operations jobs can
reload it by digest, page through findings, and require a recent `passed` report
before cleanup.

## 1. Run and record an audit

```bash
lancedb-robotics lineage audit --lake s3://bucket/robot.lance --record
```

```python
report = lake.lineage.audit()
entry = lake.lineage.record_audit_report(report, created_by="release-bot")
```

The report status is `failed` when unresolved references, missing sources, missing
table versions, or stale reversible external URNs are found. Retained versions,
retention holds, and cleanup candidates are informational findings.

Remote object-store source validation is opt-in:

```bash
lancedb-robotics lineage audit --lake s3://bucket/robot.lance \
  --check-remote-sources --record
```

Without that flag, remote validators are reported as skipped with install and
credential guidance; local path/file checks still run by default.

## 2. List and reload reports

```bash
lancedb-robotics lineage audit-reports --lake s3://bucket/robot.lance --status failed
lancedb-robotics lineage get-audit-report <report_id> --lake s3://bucket/robot.lance
```

```python
page = lake.lineage.audit_reports(status="passed", page_size=20)
entry, payload = lake.lineage.get_audit_report(page.reports[0].report_id)
```

Report identity is the content digest of the audit payload plus the lineage graph
snapshot, so the same lake state records to the same id.

## 3. Export findings

```bash
lancedb-robotics lineage export-audit-findings <report_id> \
  --lake s3://bucket/robot.lance --finding-type missing_sources --format ndjson
```

```python
findings = lake.lineage.audit_findings(report_id, finding_type="cleanup", page_size=100)
for line in lake.lineage.iter_audit_findings_ndjson(report_id, include_summary=True):
    sink.write(line)
```

Use `--page-size` / `--page-token` for JSON pages, or NDJSON for streaming
worker-friendly exports.

## 4. Gate cleanup on a recent passed audit

```bash
lancedb-robotics lake maintain --lake s3://bucket/robot.lance \
  --require-recent-audit --audit-max-age-hours 24
```

```python
from datetime import timedelta

maintain_lake(lake, require_recent_audit=True, audit_max_age=timedelta(hours=24))
```

The gate reads `lineage_audit_reports`; it does not re-run the audit inside
maintenance. If the newest whole-lake passed report is missing or stale,
maintenance refuses cleanup and tells the operator to record a new audit first.

## What's next

The v1 catalog stores report bodies inline and computes audits in-process. The
follow-on scale work is chunked/offloaded report bodies, true pushdown execution
over very large graphs, and provider-grade validator plugins for object stores
and metadata systems.
