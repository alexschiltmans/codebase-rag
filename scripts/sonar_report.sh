#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TOKEN_FILE="$PROJECT_DIR/.sonar-token"
REPORT_FILE="$PROJECT_DIR/sonar-report.md"
SONAR_URL="http://localhost:9000"
PROJECT_KEY="codebase-rag"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Read token
if [ -n "$SONAR_TOKEN" ]; then
    TOKEN="$SONAR_TOKEN"
elif [ -f "$TOKEN_FILE" ]; then
    TOKEN=$(cat "$TOKEN_FILE" | tr -d '[:space:]')
else
    echo "No token found. Run 'make sonar-start' first."
    exit 1
fi

# Wait for analysis to finish
echo "Waiting for analysis to complete..."
for i in $(seq 1 30); do
    curl -s -H "Authorization: Bearer $TOKEN" "$SONAR_URL/api/ce/component?component=$PROJECT_KEY" > "$TMP_DIR/ce.json"
    QUEUE_LEN=$(python3 -c "import json; print(len(json.load(open('$TMP_DIR/ce.json')).get('queue',[])))" 2>/dev/null || echo "0")
    if [ "$QUEUE_LEN" = "0" ]; then
        echo "Analysis complete."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Timeout waiting for analysis."
        exit 1
    fi
    sleep 2
done

# Fetch data to temp files
echo "Fetching results..."
curl -s -H "Authorization: Bearer $TOKEN" "$SONAR_URL/api/qualitygates/project_status?projectKey=$PROJECT_KEY" > "$TMP_DIR/qg.json"

METRICS="bugs,vulnerabilities,code_smells,coverage,duplicated_lines_density,ncloc,security_hotspots,reliability_rating,security_rating,sqale_rating"
curl -s -H "Authorization: Bearer $TOKEN" "$SONAR_URL/api/measures/component?component=$PROJECT_KEY&metricKeys=$METRICS" > "$TMP_DIR/measures.json"

# Paginate issues (API returns max 500 per page)
PAGE=1
echo '[]' > "$TMP_DIR/all_issues.json"
while true; do
    curl -s -H "Authorization: Bearer $TOKEN" "$SONAR_URL/api/issues/search?componentKeys=$PROJECT_KEY&statuses=OPEN,CONFIRMED,REOPENED&ps=500&p=$PAGE" > "$TMP_DIR/issues_page.json"
    python3 -c "
import json
page = json.load(open('$TMP_DIR/issues_page.json'))
all_issues = json.load(open('$TMP_DIR/all_issues.json'))
all_issues.extend(page.get('issues', []))
json.dump(all_issues, open('$TMP_DIR/all_issues.json', 'w'))
total = page.get('paging', {}).get('total', 0)
fetched = page.get('paging', {}).get('pageIndex', 1) * page.get('paging', {}).get('pageSize', 500)
print('DONE' if fetched >= total else 'MORE')
" > "$TMP_DIR/page_status.txt"
    STATUS=$(cat "$TMP_DIR/page_status.txt")
    if [ "$STATUS" = "DONE" ]; then
        break
    fi
    PAGE=$((PAGE + 1))
done

# Generate report via Python (reads from temp files, no shell interpolation issues)
python3 - "$TMP_DIR" "$REPORT_FILE" "$SONAR_URL" "$PROJECT_KEY" <<'PYEOF'
import json, sys, os
from datetime import datetime

tmp_dir, report_file, sonar_url, project_key = sys.argv[1:5]

qg = json.load(open(os.path.join(tmp_dir, "qg.json")))
measures = json.load(open(os.path.join(tmp_dir, "measures.json")))
issue_list = json.load(open(os.path.join(tmp_dir, "all_issues.json")))

qg_status = qg.get("projectStatus", {}).get("status", "UNKNOWN")
qg_icon = "✅" if qg_status == "OK" else "❌"

metric_map = {}
for m in measures.get("component", {}).get("measures", []):
    metric_map[m["metric"]] = m.get("value", "N/A")

by_type = {}
by_severity = {}
for iss in issue_list:
    t = iss.get("type", "UNKNOWN")
    s = iss.get("severity", "UNKNOWN")
    by_type[t] = by_type.get(t, 0) + 1
    by_severity[s] = by_severity.get(s, 0) + 1

lines = []
lines.append("# SonarQube Report — codebase-rag")
lines.append("")
lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append(f"**Quality Gate:** {qg_status} {qg_icon} ")
lines.append("")
lines.append("## Metrics")
lines.append("")
lines.append("| Metric | Value |")
lines.append("|--------|-------|")

labels = {
    "ncloc": "Lines of Code",
    "bugs": "Bugs",
    "vulnerabilities": "Vulnerabilities",
    "code_smells": "Code Smells",
    "security_hotspots": "Security Hotspots",
    "coverage": "Coverage (%)",
    "duplicated_lines_density": "Duplicated Lines (%)",
}
rating_map = {"1.0": "A", "2.0": "B", "3.0": "C", "4.0": "D", "5.0": "E"}
rating_labels = {
    "reliability_rating": "Reliability Rating",
    "security_rating": "Security Rating",
    "sqale_rating": "Maintainability Rating",
}

for key, label in labels.items():
    val = metric_map.get(key, "N/A")
    lines.append(f"| {label} | {val} |")
for key, label in rating_labels.items():
    val = metric_map.get(key, "N/A")
    val = rating_map.get(val, val)
    lines.append(f"| {label} | {val} |")

lines.append("")
lines.append(f"## Issues ({len(issue_list)} total)")
lines.append("")

if by_type:
    lines.append("**By type:**")
    type_labels = {"BUG": "Bug", "VULNERABILITY": "Vulnerability", "CODE_SMELL": "Code Smell", "SECURITY_HOTSPOT": "Security Hotspot"}
    for t, count in sorted(by_type.items()):
        lines.append(f"- {type_labels.get(t, t)}: {count}")
    lines.append("")

if by_severity:
    lines.append("**By severity:**")
    for s in ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]:
        if s in by_severity:
            lines.append(f"- {s.title()}: {by_severity[s]}")
    lines.append("")

if issue_list:
    lines.append("## All Issues")
    lines.append("")
    lines.append("| Severity | Type | File | Line | Message |")
    lines.append("|----------|------|------|------|---------|")
    severity_order = {"BLOCKER": 0, "CRITICAL": 1, "MAJOR": 2, "MINOR": 3, "INFO": 4}
    sorted_issues = sorted(issue_list, key=lambda x: severity_order.get(x.get("severity", "INFO"), 5))
    for iss in sorted_issues:
        sev = iss.get("severity", "?")
        typ = iss.get("type", "?")
        comp = iss.get("component", "").replace("codebase-rag:", "")
        line_no = iss.get("line", "-")
        msg = iss.get("message", "").replace("|", "\\|")
        lines.append(f"| {sev} | {typ} | {comp} | {line_no} | {msg} |")

lines.append("")

with open(report_file, "w") as f:
    f.write("\n".join(lines))
PYEOF

echo ""
echo "=== Report written to sonar-report.md ==="
echo ""
cat "$REPORT_FILE"
