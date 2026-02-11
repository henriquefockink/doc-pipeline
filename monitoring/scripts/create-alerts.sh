#!/bin/bash
# Script para criar/atualizar TODOS os alertas do Doc Pipeline no Grafana via API
#
# Este script eh a SOURCE OF TRUTH para todos os alertas.
# Ele usa DELETE + POST para garantir que os alertas ficam atualizados.
#
# Uso: ./create-alerts.sh [grafana_url] [api_token] [datasource_uid]
#
# Variaveis de ambiente (ou .env):
#   GRAFANA_URL   - URL do Grafana
#   GRAFANA_TOKEN - Token de API (glsa_xxx) ou user:password
#
# Exemplos:
#   ./create-alerts.sh                                          # usa .env
#   ./create-alerts.sh https://grafana.exemplo.com glsa_xxx     # passa como argumento

set -e

# Carrega .env se existir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
elif [ -f ".env" ]; then
    source ".env"
fi

# Parametros: argumentos > variaveis de ambiente > defaults
GRAFANA_URL="${1:-${GRAFANA_URL:-http://localhost:3000}}"
AUTH="${2:-${GRAFANA_TOKEN:-admin:admin}}"
DS_UID_OVERRIDE="${3:-}"

# Remove trailing slash
GRAFANA_URL="${GRAFANA_URL%/}"

# Detecta tipo de auth (token ou user:pass)
if [[ "$AUTH" == glsa_* ]] || [[ "$AUTH" == sa-* ]]; then
    AUTH_HEADER="Authorization: Bearer $AUTH"
else
    AUTH_HEADER="Authorization: Basic $(echo -n "$AUTH" | base64)"
fi

echo "Grafana URL: $GRAFANA_URL"

# Busca UID do datasource Prometheus
if [ -n "$DS_UID_OVERRIDE" ]; then
    DS_UID="$DS_UID_OVERRIDE"
    echo "Datasource UID (override): $DS_UID"
else
    echo -n "Buscando datasource Prometheus... "
    DS_RESPONSE=$(curl -s "$GRAFANA_URL/api/datasources" -H "$AUTH_HEADER")
    DS_UID=$(echo "$DS_RESPONSE" | grep -o '"uid":"[^"]*","orgId":[0-9]*,"name":"[^"]*","type":"prometheus"' | head -1 | grep -o '"uid":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$DS_UID" ]; then
        echo "ERRO: Prometheus datasource nao encontrado!"
        echo "Datasources disponiveis:"
        echo "$DS_RESPONSE" | grep -o '"name":"[^"]*","type":"[^"]*"' | sed 's/"name":"//g; s/","type":"/ -> /g; s/"//g'
        exit 1
    fi
    echo "$DS_UID"
fi

# Cria folder 'Doc Pipeline' se nao existe
echo -n "Criando folder 'Doc Pipeline'... "
FOLDER_RESPONSE=$(curl -s -X POST "$GRAFANA_URL/api/folders" \
    -H "$AUTH_HEADER" \
    -H "Content-Type: application/json" \
    -d '{"title": "Doc Pipeline"}' 2>/dev/null || echo '{}')

FOLDER_UID=$(echo "$FOLDER_RESPONSE" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$FOLDER_UID" ]; then
    FOLDER_UID=$(curl -s "$GRAFANA_URL/api/folders" \
        -H "$AUTH_HEADER" | grep -o '"uid":"[^"]*","title":"Doc Pipeline"' | cut -d'"' -f4)
fi

if [ -z "$FOLDER_UID" ]; then
    echo "ERRO: Nao foi possivel criar/encontrar folder"
    exit 1
fi
echo "$FOLDER_UID"

# Cria folder aninhado 'Workers Alerts' dentro de 'Doc Pipeline'
echo -n "Criando folder 'Workers Alerts' (aninhado)... "
WORKERS_ALERTS_RESPONSE=$(curl -s -X POST "$GRAFANA_URL/api/folders" \
    -H "$AUTH_HEADER" \
    -H "Content-Type: application/json" \
    -d '{"title": "Workers Alerts", "uid": "doc-pipeline-workers-alerts-nested", "parentUid": "'"$FOLDER_UID"'"}' 2>/dev/null || echo '{}')

WORKERS_ALERTS_UID=$(echo "$WORKERS_ALERTS_RESPONSE" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$WORKERS_ALERTS_UID" ]; then
    WORKERS_ALERTS_UID="doc-pipeline-workers-alerts-nested"
fi
echo "$WORKERS_ALERTS_UID"

# ============================================================
# Helper: cria alerta com pattern correto A->B(reduce)->C(threshold)
# Para alertas simples: 1 query PromQL + reduce + threshold
# ============================================================
create_alert_simple() {
    local uid="$1"
    local title="$2"
    local folder="$3"
    local expr="$4"
    local threshold_type="$5"   # gt ou lt
    local threshold_val="$6"
    local for_dur="$7"
    local severity="$8"
    local summary="$9"
    local desc="${10}"
    local extra_labels="${11:-}"
    local time_range="${12:-300}"
    local no_data="${13:-OK}"

    local labels='"severity":"'$severity'","service":"doc-pipeline"'
    [ -n "$extra_labels" ] && labels="$labels,$extra_labels"

    # Delete existing (ignore errors)
    curl -s -X DELETE "$GRAFANA_URL/api/v1/provisioning/alert-rules/$uid" \
        -H "$AUTH_HEADER" > /dev/null 2>&1 || true

    echo -n "  $title... "
    RESP=$(curl -s -w "%{http_code}" -X POST "$GRAFANA_URL/api/v1/provisioning/alert-rules" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -H "X-Disable-Provenance: true" \
        -d '{
            "uid": "'"$uid"'",
            "title": "'"$title"'",
            "ruleGroup": "doc-pipeline-alerts",
            "folderUID": "'"$folder"'",
            "condition": "C",
            "data": [
                {"refId":"A","relativeTimeRange":{"from":'"$time_range"',"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"'"$expr"'","intervalMs":1000,"maxDataPoints":43200}},
                {"refId":"B","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"A","settings":{"mode":"dropNN"}}},
                {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"B","conditions":[{"evaluator":{"type":"'"$threshold_type"'","params":['"$threshold_val"']}}]}}
            ],
            "noDataState": "'"$no_data"'",
            "execErrState": "Error",
            "for": "'"$for_dur"'",
            "annotations": {"summary":"'"$summary"'","description":"'"$desc"'"},
            "labels": {'"$labels"'}
        }')

    CODE="${RESP: -3}"
    if [ "$CODE" = "201" ]; then
        echo "OK"
    else
        echo "FAIL ($CODE)"
    fi
}

# ============================================================
# Helper: cria alerta com math expression (2+ queries)
# Pattern: A,B -> math(C) -> reduce(D) -> threshold(E)
# ============================================================
create_alert_math() {
    local uid="$1"
    local title="$2"
    local folder="$3"
    local data_json="$4"
    local condition="$5"
    local for_dur="$6"
    local severity="$7"
    local summary="$8"
    local desc="$9"
    local extra_labels="${10:-}"
    local no_data="${11:-OK}"

    local labels='"severity":"'$severity'","service":"doc-pipeline"'
    [ -n "$extra_labels" ] && labels="$labels,$extra_labels"

    # Delete existing
    curl -s -X DELETE "$GRAFANA_URL/api/v1/provisioning/alert-rules/$uid" \
        -H "$AUTH_HEADER" > /dev/null 2>&1 || true

    echo -n "  $title... "
    RESP=$(curl -s -w "%{http_code}" -X POST "$GRAFANA_URL/api/v1/provisioning/alert-rules" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -H "X-Disable-Provenance: true" \
        -d '{
            "uid": "'"$uid"'",
            "title": "'"$title"'",
            "ruleGroup": "doc-pipeline-alerts",
            "folderUID": "'"$folder"'",
            "condition": "'"$condition"'",
            "data": '"$data_json"',
            "noDataState": "'"$no_data"'",
            "execErrState": "Error",
            "for": "'"$for_dur"'",
            "annotations": {"summary":"'"$summary"'","description":"'"$desc"'"},
            "labels": {'"$labels"'}
        }')

    CODE="${RESP: -3}"
    if [ "$CODE" = "201" ]; then
        echo "OK"
    else
        echo "FAIL ($CODE)"
    fi
}

echo ""
echo "============================================"
echo "Criando alertas de API..."
echo "============================================"

# 1. High Error Rate (5xx) — math: A/B -> reduce -> threshold
create_alert_math "dp-high-error-rate" "High Error Rate (5xx)" "$FOLDER_UID" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{status=~\"5..\",endpoint=~\"/classify|/extract|/process\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=~\"/classify|/extract|/process\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"$A / $B"}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"C","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"D","conditions":[{"evaluator":{"type":"gt","params":[0.05]}}]}}
]' "E" "5m" "critical" "High 5xx error rate" "Error rate above 5%"

# 2. High Latency P95
create_alert_simple "dp-high-latency-p95" "High Latency P95" "$FOLDER_UID" \
    'histogram_quantile(0.95,sum(rate(doc_pipeline_request_duration_seconds_bucket{endpoint=~\"/classify|/extract|/process\"}[5m])) by (le))' \
    "gt" "30" "5m" "warning" "High P95 latency" "P95 latency above 30s"

# 3. Critical Latency P99
create_alert_simple "dp-critical-latency-p99" "Critical Latency P99" "$FOLDER_UID" \
    'histogram_quantile(0.99,sum(rate(doc_pipeline_request_duration_seconds_bucket{endpoint=~\"/classify|/extract|/process\"}[5m])) by (le))' \
    "gt" "60" "5m" "critical" "Critical P99 latency" "P99 latency above 60s"

# 4. High Request Concurrency
create_alert_simple "dp-high-concurrency" "High Request Concurrency" "$FOLDER_UID" \
    'sum(doc_pipeline_requests_in_progress{endpoint=~\"/classify|/extract|/process\"})' \
    "gt" "10" "5m" "warning" "High concurrency" "More than 10 concurrent requests"

# 5. Low DocID Confidence
create_alert_simple "dp-low-confidence" "Low DocID Confidence" "$FOLDER_UID" \
    'histogram_quantile(0.5,sum(rate(doc_pipeline_classification_confidence_bucket[10m])) by (le))' \
    "lt" "0.7" "15m" "warning" "Low classification confidence" "Median confidence below 0.7" "" "600"

# 6. Very Low DocID Confidence
create_alert_simple "dp-very-low-confidence" "Very Low DocID Confidence" "$FOLDER_UID" \
    'histogram_quantile(0.5,sum(rate(doc_pipeline_classification_confidence_bucket[5m])) by (le))' \
    "lt" "0.5" "5m" "critical" "Very low confidence" "Median confidence below 0.5 - model degraded"

# 7. /classify Endpoint Errors — math: A/B -> reduce -> threshold
create_alert_math "dp-classify-errors" "/classify Endpoint Errors" "$FOLDER_UID" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/classify\",status=~\"5..\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/classify\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"$A / $B"}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"C","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"D","conditions":[{"evaluator":{"type":"gt","params":[0.05]}}]}}
]' "E" "5m" "critical" "High error rate on /classify" "Classification error rate above 5%" '"endpoint":"classify"'

# 8. /extract Endpoint Errors
create_alert_math "dp-extract-errors" "/extract Endpoint Errors" "$FOLDER_UID" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/extract\",status=~\"5..\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/extract\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"$A / $B"}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"C","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"D","conditions":[{"evaluator":{"type":"gt","params":[0.05]}}]}}
]' "E" "5m" "critical" "High error rate on /extract" "Extraction error rate above 5%" '"endpoint":"extract"'

# 9. /process Endpoint Errors
create_alert_math "dp-process-errors" "/process Endpoint Errors" "$FOLDER_UID" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/process\",status=~\"5..\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/process\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"$A / $B"}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"C","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"D","conditions":[{"evaluator":{"type":"gt","params":[0.05]}}]}}
]' "E" "5m" "critical" "High error rate on /process" "Full pipeline error rate above 5%" '"endpoint":"process"'

echo ""
echo "============================================"
echo "Criando alertas de Queue & Worker..."
echo "============================================"

# 10. High Queue Depth
create_alert_simple "dp-queue-depth-high" "High Queue Depth" "$FOLDER_UID" \
    'max(doc_pipeline_queue_depth)' \
    "gt" "50" "5m" "warning" "Queue depth is high" "Jobs waiting in queue above 50" '"component":"queue"'

# 11. Critical Queue Depth
create_alert_simple "dp-queue-depth-critical" "Critical Queue Depth" "$FOLDER_UID" \
    'max(doc_pipeline_queue_depth)' \
    "gt" "90" "2m" "critical" "Queue depth critical" "Queue near capacity (>90 jobs)" '"component":"queue"'

# 12. High Queue Wait Time
create_alert_simple "dp-queue-wait-high" "High Queue Wait Time" "$FOLDER_UID" \
    'histogram_quantile(0.95,sum(rate(doc_pipeline_queue_wait_seconds_bucket[5m])) by (le))' \
    "gt" "60" "5m" "warning" "Jobs waiting too long" "P95 queue wait time above 60s" '"component":"queue"'

# 13. Worker Not Processing — math: queue>0 && rate==0
create_alert_math "dp-worker-not-processing" "Worker Not Processing" "$FOLDER_UID" '[
    {"refId":"A","relativeTimeRange":{"from":600,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"max(doc_pipeline_queue_depth)","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":600,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"A","settings":{"mode":"dropNN"}}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"B","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"($C > 0) && ($D == 0)"}},
    {"refId":"F","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"E","conditions":[{"evaluator":{"type":"gt","params":[0]}}]}}
]' "F" "5m" "critical" "Worker not processing jobs" "Queue has jobs but worker is not processing" '"component":"worker"'

# 14. Worker Error Rate — math: errors/total -> reduce -> threshold
create_alert_math "dp-worker-error-rate" "Worker Error Rate" "$FOLDER_UID" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total{status=\"error\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"$A / $B"}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"C","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"D","conditions":[{"evaluator":{"type":"gt","params":[0.1]}}]}}
]' "E" "5m" "critical" "High worker error rate" "More than 10% of jobs failing" '"component":"worker"'

# 15. Webhook Failures
create_alert_simple "dp-webhook-failures" "Webhook Failures" "$FOLDER_UID" \
    'sum(rate(doc_pipeline_webhook_deliveries_total{status=\"failed\"}[5m]))' \
    "gt" "0.1" "5m" "warning" "Webhook deliveries failing" "Webhooks failing to deliver" '"component":"webhook"'

echo ""
echo "============================================"
echo "Criando alertas de Workers (folder aninhado)..."
echo "============================================"

# ============================================================
# DocID Worker Alerts
# ============================================================

# 16. DocID Worker Down (all workers dead — uses Redis heartbeat TTL=60s)
# noDataState=Alerting because when all workers die, metric disappears entirely
create_alert_simple "dp-docid-worker-down" "DocID Worker Down" "$WORKERS_ALERTS_UID" \
    'max(doc_pipeline_worker_up{worker_id=~\"docid.*\"}) or vector(0)' \
    "lt" "1" "2m" "critical" "DocID Worker is down" "All DocID workers are unreachable" '"worker":"docid"' "300" "Alerting"

# 16b. DocID Workers Degraded (fewer than 5 workers alive)
create_alert_simple "dp-docid-workers-degraded" "DocID Workers Degraded" "$WORKERS_ALERTS_UID" \
    'count(doc_pipeline_worker_up{worker_id=~\"docid.*\"} == 1) or vector(0)' \
    "lt" "5" "5m" "warning" "DocID workers degraded" "Fewer than 5 DocID workers are running - Docker failed to restart" '"worker":"docid"' "300" "Alerting"

# 17. DocID High Error Rate
create_alert_math "dp-docid-error-rate" "DocID High Error Rate" "$WORKERS_ALERTS_UID" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-api\",status=\"error\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-api\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"$A / $B"}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"C","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"D","conditions":[{"evaluator":{"type":"gt","params":[0.1]}}]}}
]' "E" "5m" "warning" "High classification worker error rate" "Classification worker error rate above 10%" '"worker":"docid"'

# 18. DocID High Latency
create_alert_simple "dp-docid-high-latency" "DocID High Latency" "$WORKERS_ALERTS_UID" \
    'histogram_quantile(0.95,sum(rate(doc_pipeline_worker_processing_seconds_bucket{job=\"doc-pipeline-api\",worker_id=~\"docid.*\"}[5m])) by (le))' \
    "gt" "30" "5m" "warning" "Classification processing time is high" "P95 processing time above 30 seconds" '"worker":"docid"'

# 19. DocID Queue Backup
create_alert_simple "dp-docid-queue-backup" "DocID Queue Backup" "$WORKERS_ALERTS_UID" \
    'max(doc_pipeline_queue_depth{job=\"doc-pipeline-api\"})' \
    "gt" "10" "5m" "warning" "Classification queue is backing up" "Queue depth above 10 jobs" '"worker":"docid","component":"queue"'

# 20. DocID Low Confidence
create_alert_simple "dp-docid-low-confidence" "DocID Low Confidence" "$WORKERS_ALERTS_UID" \
    'histogram_quantile(0.5,sum(rate(doc_pipeline_classification_confidence_bucket[10m])) by (le))' \
    "lt" "0.7" "10m" "warning" "Classification confidence is low" "Median classification confidence below 70%" '"worker":"docid"' "600"

# ============================================================
# OCR Worker Alerts
# ============================================================

# 21. OCR Worker Down
# noDataState=Alerting because if target disappears, worker is definitely down
create_alert_simple "dp-ocr-worker-down" "OCR Worker Down" "$WORKERS_ALERTS_UID" \
    'up{job=\"doc-pipeline-worker-ocr\"}' \
    "lt" "1" "2m" "critical" "OCR Worker is down" "The OCR worker has been unreachable for more than 2 minutes" '"worker":"ocr"' "300" "Alerting"

# 22. OCR High Error Rate
create_alert_math "dp-ocr-error-rate" "OCR High Error Rate" "$WORKERS_ALERTS_UID" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total{operation=\"ocr\",status=\"error\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total{operation=\"ocr\"}[5m]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"$A / $B"}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"C","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"D","conditions":[{"evaluator":{"type":"gt","params":[0.1]}}]}}
]' "E" "5m" "warning" "High OCR error rate" "OCR error rate above 10%" '"worker":"ocr"'

# 23. OCR High Latency
create_alert_simple "dp-ocr-high-latency" "OCR High Latency" "$WORKERS_ALERTS_UID" \
    'histogram_quantile(0.95,sum(rate(doc_pipeline_worker_processing_seconds_bucket{operation=\"ocr\"}[5m])) by (le))' \
    "gt" "30" "5m" "warning" "OCR processing time is high" "P95 processing time above 30 seconds" '"worker":"ocr"'

# 24. OCR Queue Backup
create_alert_simple "dp-ocr-queue-backup" "OCR Queue Backup" "$WORKERS_ALERTS_UID" \
    'doc_pipeline_queue_depth{job=\"doc-pipeline-worker-ocr\"}' \
    "gt" "10" "5m" "warning" "OCR queue is backing up" "Queue depth above 10 jobs" '"worker":"ocr","component":"queue"'

echo ""
echo "============================================"
echo "Criando alertas de SLO..."
echo "============================================"

# 25. SLO: Availability Below 99% — fixed: when total=0, result=1 (no alarm)
create_alert_math "dp-slo-availability" "SLO: Availability Below 99%" "$FOLDER_UID" '[
    {"refId":"A","relativeTimeRange":{"from":3600,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{status=~\"5..\",endpoint=~\"/classify|/extract|/process\"}[1h]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":3600,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=~\"/classify|/extract|/process\"}[1h]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"A","settings":{"mode":"dropNN"}}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"B","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"($D > 0) * (1 - ($C / $D)) + ($D == 0) * 1"}},
    {"refId":"F","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"E","conditions":[{"evaluator":{"type":"lt","params":[0.99]}}]}}
]' "F" "5m" "critical" "SLO: Availability below 99%" "Service availability is below 99% target" '"slo":"availability"'

# 26. SLO: Latency Below Target — fixed: when total=0, result=1 (no alarm)
create_alert_math "dp-slo-latency" "SLO: Latency Below Target" "$FOLDER_UID" '[
    {"refId":"A","relativeTimeRange":{"from":3600,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_request_duration_seconds_bucket{le=\"30\",endpoint=~\"/classify|/extract|/process\"}[1h]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"B","relativeTimeRange":{"from":3600,"to":0},"datasourceUid":"'"$DS_UID"'","model":{"expr":"sum(rate(doc_pipeline_request_duration_seconds_count{endpoint=~\"/classify|/extract|/process\"}[1h]))","intervalMs":1000,"maxDataPoints":43200}},
    {"refId":"C","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"A","settings":{"mode":"dropNN"}}},
    {"refId":"D","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"reduce","reducer":"last","expression":"B","settings":{"mode":"dropNN"}}},
    {"refId":"E","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"math","expression":"($D > 0) * ($C / $D) + ($D == 0) * 1"}},
    {"refId":"F","relativeTimeRange":{"from":0,"to":0},"datasourceUid":"__expr__","model":{"type":"threshold","expression":"E","conditions":[{"evaluator":{"type":"lt","params":[0.95]}}]}}
]' "F" "5m" "warning" "SLO: Less than 95% of requests under 30s" "Latency SLO not met" '"slo":"latency"'

echo ""
echo "============================================"
echo "Concluido! Total: 26 alertas"
echo "Verifique em: $GRAFANA_URL/alerting/list"
echo "============================================"
