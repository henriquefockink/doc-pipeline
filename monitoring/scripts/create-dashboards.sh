#!/bin/bash
# Script para criar dashboards no Grafana via API
#
# Uso: ./create-dashboards.sh [grafana_url] [api_token]
#
# Variáveis de ambiente (ou .env):
#   GRAFANA_URL   - URL do Grafana
#   GRAFANA_TOKEN - Token de API (glsa_xxx) ou user:password

set -e

# Carrega .env se existir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
elif [ -f ".env" ]; then
    source ".env"
fi

# Parâmetros
GRAFANA_URL="${1:-${GRAFANA_URL:-http://localhost:3000}}"
AUTH="${2:-${GRAFANA_TOKEN:-admin:admin}}"
GRAFANA_URL="${GRAFANA_URL%/}"

# Auth header
if [[ "$AUTH" == glsa_* ]] || [[ "$AUTH" == sa-* ]]; then
    AUTH_HEADER="Authorization: Bearer $AUTH"
else
    AUTH_HEADER="Authorization: Basic $(echo -n "$AUTH" | base64)"
fi

echo "Grafana URL: $GRAFANA_URL"

# Busca UID do datasource Prometheus
echo -n "Buscando datasource Prometheus... "
DS_RESPONSE=$(curl -s "$GRAFANA_URL/api/datasources" -H "$AUTH_HEADER")
DS_UID=$(echo "$DS_RESPONSE" | grep -o '"uid":"[^"]*","orgId":[0-9]*,"name":"[^"]*","type":"prometheus"' | head -1 | grep -o '"uid":"[^"]*"' | cut -d'"' -f4)

if [ -z "$DS_UID" ]; then
    echo "ERRO: Prometheus datasource não encontrado!"
    exit 1
fi
echo "$DS_UID"

# Usa folders existentes
FOLDER_MAIN="bfbjyfdf0uhhcf"
FOLDER_WORKERS="doc-pipeline-workers-nested"
echo "Usando folders existentes:"
echo "  Doc Pipeline: $FOLDER_MAIN"
echo "  Workers: $FOLDER_WORKERS"

# Função para criar/atualizar dashboard
create_dashboard() {
    local title="$1"
    local uid="$2"
    local folder_uid="$3"
    local dashboard_json="$4"

    echo -n "  $title... "

    # Substitui datasource UID no JSON
    dashboard_json=$(echo "$dashboard_json" | sed "s/\${datasource}/$DS_UID/g")

    RESP=$(curl -s -w "%{http_code}" -X POST "$GRAFANA_URL/api/dashboards/db" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -d '{
            "dashboard": '"$dashboard_json"',
            "folderUid": "'"$folder_uid"'",
            "overwrite": true
        }')

    CODE="${RESP: -3}"

    if [ "$CODE" = "200" ]; then
        echo "OK (updated)"
    elif [ "$CODE" = "201" ]; then
        echo "OK (created)"
    else
        echo "FAIL ($CODE)"
        echo "$RESP" | head -c 200
        echo ""
    fi
}

echo ""
echo "Criando dashboards..."

# ============================================================
# Dashboard Principal: Doc Pipeline
# ============================================================
create_dashboard "Doc Pipeline" "doc-pipeline" "$FOLDER_MAIN" '{
  "uid": "doc-pipeline",
  "title": "Doc Pipeline",
  "description": "Dashboard consolidado do pipeline de documentos",
  "tags": ["doc-pipeline"],
  "timezone": "browser",
  "refresh": "30s",
  "time": {"from": "now-1h", "to": "now"},
  "panels": [

    {"type": "row", "title": "⚡ Workers & Queue", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}, "collapsed": false},

    {"id": 7, "type": "stat", "title": "Workers Ativos", "gridPos": {"h": 8, "w": 6, "x": 0, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {
        "defaults": {
          "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": null}, {"color": "yellow", "value": 3}, {"color": "green", "value": 5}]},
          "color": {"mode": "thresholds"},
          "unit": "none", "min": 0, "max": 5
        },
        "overrides": []
      },
      "options": {"colorMode": "value", "graphMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}, "textMode": "value", "text": {"titleSize": 20, "valueSize": 72}},
      "targets": [{"expr": "count(doc_pipeline_worker_up{worker_id=~\"docid.*\"} == 1) or vector(0)", "legendFormat": "Workers"}]},

    {"id": 9, "type": "timeseries", "title": "Workers Ativos (historico)", "gridPos": {"h": 8, "w": 10, "x": 6, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {
        "defaults": {"color": {"mode": "fixed", "fixedColor": "blue"}, "min": 0, "max": 6, "custom": {"drawStyle": "line", "fillOpacity": 20}}
      },
      "options": {"legend": {"calcs": ["lastNotNull"], "displayMode": "list", "placement": "bottom"}},
      "targets": [
        {"expr": "count(doc_pipeline_worker_up{worker_id=~\"docid.*\"} == 1) or vector(0)", "legendFormat": "DocID Workers"}
      ]},

    {"id": 1, "type": "stat", "title": "Queue", "gridPos": {"h": 4, "w": 4, "x": 16, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 5}, {"color": "red", "value": 15}]}, "unit": "none"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "max(doc_pipeline_queue_depth) or vector(0)", "legendFormat": "Queue"}]},

    {"id": 8, "type": "stat", "title": "Confidence", "gridPos": {"h": 4, "w": 4, "x": 20, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": null}, {"color": "yellow", "value": 0.7}, {"color": "green", "value": 0.9}]}, "unit": "percentunit", "min": 0, "max": 1}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "histogram_quantile(0.5, sum(rate(doc_pipeline_classification_confidence_bucket[5m])) by (le))", "legendFormat": "Median"}]},

    {"id": 4, "type": "stat", "title": "Latency P95", "gridPos": {"h": 4, "w": 4, "x": 16, "y": 5},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 10}, {"color": "red", "value": 30}]}, "unit": "s"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "histogram_quantile(0.95, sum(rate(doc_pipeline_request_duration_seconds_bucket[5m])) by (le))", "legendFormat": "P95"}]},

    {"id": 5, "type": "stat", "title": "Error Rate", "gridPos": {"h": 4, "w": 4, "x": 20, "y": 5},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 0.01}, {"color": "red", "value": 0.05}]}, "unit": "percentunit", "max": 1}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(rate(doc_pipeline_requests_total{status=~\"4..|5..\"}[5m])) / sum(rate(doc_pipeline_requests_total[5m])) or vector(0)", "legendFormat": "Error Rate"}]},

    {"type": "row", "title": "📊 Overview", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 9}, "collapsed": false},

    {"id": 2, "type": "stat", "title": "Req/min", "gridPos": {"h": 4, "w": 4, "x": 0, "y": 10},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short", "decimals": 1}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(rate(doc_pipeline_requests_total[5m])) * 60", "legendFormat": "Req/min"}]},

    {"id": 3, "type": "stat", "title": "Jobs (período)", "gridPos": {"h": 4, "w": 4, "x": 4, "y": 10},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(increase(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-api\"}[$__range]))", "legendFormat": "Jobs"}]},

    {"id": 6, "type": "stat", "title": "Erros (período)", "gridPos": {"h": 4, "w": 4, "x": 8, "y": 10},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 1}, {"color": "red", "value": 5}]}, "unit": "short"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(increase(doc_pipeline_jobs_processed_total{status=\"error\"}[$__range])) or vector(0)", "legendFormat": "Erros"}]},

    {"id": 10, "type": "timeseries", "title": "Queue vs Workers", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 10},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "custom": {"drawStyle": "line", "fillOpacity": 10, "axisSoftMin": 0}}},
      "options": {"legend": {"calcs": ["lastNotNull", "max"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "max(doc_pipeline_queue_depth) or vector(0)", "legendFormat": "Queue Depth"},
        {"expr": "count(doc_pipeline_worker_up{worker_id=~\"docid.*\"} == 1) or vector(0)", "legendFormat": "Workers"}
      ]},

    {"id": 11, "type": "timeseries", "title": "Queue Wait Time", "gridPos": {"h": 4, "w": 12, "x": 0, "y": 14},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "s", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "max"], "displayMode": "list", "placement": "right"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, sum(rate(doc_pipeline_queue_wait_seconds_bucket{job=\"doc-pipeline-api\"}[5m])) by (le))", "legendFormat": "P50"},
        {"expr": "histogram_quantile(0.95, sum(rate(doc_pipeline_queue_wait_seconds_bucket{job=\"doc-pipeline-api\"}[5m])) by (le))", "legendFormat": "P95"}
      ]},

    {"type": "row", "title": "🌐 API", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 18}, "collapsed": false},

    {"id": 20, "type": "timeseries", "title": "Request Rate por Endpoint", "gridPos": {"h": 8, "w": 8, "x": 0, "y": 19},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "reqps", "custom": {"drawStyle": "line", "fillOpacity": 20}}},
      "options": {"legend": {"calcs": ["mean", "lastNotNull"], "displayMode": "table", "placement": "bottom"}},
      "targets": [{"expr": "sum by (endpoint) (rate(doc_pipeline_requests_total[5m]))", "legendFormat": "{{endpoint}}"}]},

    {"id": 21, "type": "timeseries", "title": "Latency por Endpoint (P50/P95)", "gridPos": {"h": 8, "w": 8, "x": 8, "y": 19},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "s", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, sum by (endpoint, le) (rate(doc_pipeline_request_duration_seconds_bucket[5m])))", "legendFormat": "P50 {{endpoint}}"},
        {"expr": "histogram_quantile(0.95, sum by (endpoint, le) (rate(doc_pipeline_request_duration_seconds_bucket[5m])))", "legendFormat": "P95 {{endpoint}}"}
      ]},

    {"id": 22, "type": "timeseries", "title": "Status Codes", "gridPos": {"h": 8, "w": 8, "x": 16, "y": 19},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "reqps", "custom": {"drawStyle": "bars", "fillOpacity": 80, "stacking": {"mode": "normal"}}}},
      "options": {"legend": {"calcs": ["sum"], "displayMode": "table", "placement": "bottom"}},
      "targets": [{"expr": "sum by (status) (rate(doc_pipeline_requests_total[5m]))", "legendFormat": "{{status}}"}]},

    {"type": "row", "title": "⚙️ Processing", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 27}, "collapsed": false},

    {"id": 30, "type": "timeseries", "title": "Processing Time por Operação (P50/P95)", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 28},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "s", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, sum(rate(doc_pipeline_worker_processing_seconds_bucket{job=\"doc-pipeline-api\"}[5m])) by (le, operation))", "legendFormat": "P50 {{operation}}"},
        {"expr": "histogram_quantile(0.95, sum(rate(doc_pipeline_worker_processing_seconds_bucket{job=\"doc-pipeline-api\"}[5m])) by (le, operation))", "legendFormat": "P95 {{operation}}"}
      ]},

    {"id": 31, "type": "timeseries", "title": "Jobs por Status", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 28},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "ops", "custom": {"drawStyle": "line", "fillOpacity": 20}}},
      "options": {"legend": {"calcs": ["mean", "sum"], "displayMode": "table", "placement": "bottom"}},
      "targets": [{"expr": "sum(rate(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-api\"}[5m])) by (operation, status)", "legendFormat": "{{operation}} - {{status}}"}]},

    {"type": "row", "title": "📄 Documents", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 36}, "collapsed": false},

    {"id": 40, "type": "timeseries", "title": "Documentos por Tipo", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 37},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short", "custom": {"drawStyle": "bars", "fillOpacity": 80, "stacking": {"mode": "normal"}}}},
      "options": {"legend": {"calcs": ["sum"], "displayMode": "table", "placement": "bottom"}},
      "targets": [{"expr": "increase(doc_pipeline_documents_processed_total[1h])", "legendFormat": "{{document_type}}"}]},

    {"id": 41, "type": "timeseries", "title": "Classification Confidence", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 37},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "percentunit", "min": 0, "max": 1, "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "lastNotNull"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, sum(rate(doc_pipeline_classification_confidence_bucket[5m])) by (le))", "legendFormat": "P50"},
        {"expr": "histogram_quantile(0.95, sum(rate(doc_pipeline_classification_confidence_bucket[5m])) by (le))", "legendFormat": "P95"},
        {"expr": "histogram_quantile(0.05, sum(rate(doc_pipeline_classification_confidence_bucket[5m])) by (le))", "legendFormat": "P5 (worst)"}
      ]},

    {"type": "row", "title": "👥 Clients", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 45}, "collapsed": false},

    {"id": 50, "type": "piechart", "title": "Requests por Cliente (período)", "gridPos": {"h": 8, "w": 8, "x": 0, "y": 46},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short"}},
      "options": {"legend": {"displayMode": "table", "placement": "right", "values": ["percent", "value"]}, "pieType": "pie", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(increase(doc_pipeline_requests_by_client_total[$__range])) by (client)", "legendFormat": "{{client}}"}]},

    {"id": 51, "type": "timeseries", "title": "Requests por Cliente", "gridPos": {"h": 8, "w": 16, "x": 8, "y": 46},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "reqps", "custom": {"drawStyle": "line", "fillOpacity": 20}}},
      "options": {"legend": {"calcs": ["sum", "lastNotNull"], "displayMode": "table", "placement": "bottom"}},
      "targets": [{"expr": "sum(rate(doc_pipeline_requests_by_client_total[5m])) by (client)", "legendFormat": "{{client}}"}]},

    {"type": "row", "title": "📤 Webhooks", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 54}, "collapsed": true, "panels": [
      {"id": 60, "type": "stat", "title": "Delivered (período)", "gridPos": {"h": 4, "w": 6, "x": 0, "y": 55},
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}]}, "unit": "short"}},
        "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
        "targets": [{"expr": "sum(increase(doc_pipeline_webhook_deliveries_total{status=\"success\"}[$__range])) or vector(0)", "legendFormat": "Delivered"}]},

      {"id": 61, "type": "stat", "title": "Failed (período)", "gridPos": {"h": 4, "w": 6, "x": 6, "y": 55},
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "red", "value": 1}]}, "unit": "short"}},
        "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
        "targets": [{"expr": "sum(increase(doc_pipeline_webhook_deliveries_total{status=\"failed\"}[$__range])) or vector(0)", "legendFormat": "Failed"}]},

      {"id": 62, "type": "timeseries", "title": "Webhook Deliveries", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 55},
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "ops", "custom": {"drawStyle": "line", "fillOpacity": 20}}},
        "options": {"legend": {"calcs": ["sum"], "displayMode": "table", "placement": "bottom"}},
        "targets": [{"expr": "sum(rate(doc_pipeline_webhook_deliveries_total[5m])) by (status)", "legendFormat": "{{status}}"}]}
    ]}
  ]
}'

# ============================================================
# Dashboard: Worker OCR (separado pois é outro serviço)
# ============================================================
create_dashboard "Worker OCR" "worker-ocr" "$FOLDER_WORKERS" '{
  "uid": "worker-ocr",
  "title": "Worker OCR",
  "description": "Métricas do worker de OCR genérico",
  "tags": ["doc-pipeline", "ocr", "worker"],
  "timezone": "browser",
  "refresh": "30s",
  "time": {"from": "now-1h", "to": "now"},
  "panels": [
    {"type": "row", "title": "Overview", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}},

    {"type": "stat", "title": "Queue", "gridPos": {"h": 5, "w": 4, "x": 0, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 5}, {"color": "red", "value": 10}]}, "unit": "none"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "doc_pipeline_queue_depth{job=\"doc-pipeline-worker-ocr\"}", "legendFormat": "Queue"}]},

    {"type": "stat", "title": "Jobs (período)", "gridPos": {"h": 5, "w": 4, "x": 4, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(increase(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-worker-ocr\"}[$__range]))", "legendFormat": "Jobs"}]},

    {"type": "stat", "title": "P95 Latency", "gridPos": {"h": 5, "w": 4, "x": 8, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 5}, {"color": "red", "value": 15}]}, "unit": "s"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "histogram_quantile(0.95, sum(rate(doc_pipeline_worker_processing_seconds_bucket{job=\"doc-pipeline-worker-ocr\"}[5m])) by (le))", "legendFormat": "P95"}]},

    {"type": "stat", "title": "Error Rate", "gridPos": {"h": 5, "w": 4, "x": 12, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 0.01}, {"color": "red", "value": 0.05}]}, "unit": "percentunit"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(rate(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-worker-ocr\",status=\"error\"}[5m])) / sum(rate(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-worker-ocr\"}[5m])) or vector(0)", "legendFormat": "Error Rate"}]},

    {"type": "stat", "title": "Erros (período)", "gridPos": {"h": 5, "w": 4, "x": 16, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 1}, {"color": "red", "value": 5}]}, "unit": "short"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(increase(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-worker-ocr\",status=\"error\"}[$__range])) or vector(0)", "legendFormat": "Erros"}]},

    {"type": "stat", "title": "Worker", "gridPos": {"h": 5, "w": 4, "x": 20, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "mappings": [{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}, "1": {"text": "UP", "color": "green"}}}], "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": null}, {"color": "green", "value": 1}]}}},
      "options": {"colorMode": "background", "graphMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "up{job=\"doc-pipeline-worker-ocr\"}", "legendFormat": "Status"}]},

    {"type": "row", "title": "Processing", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 6}},

    {"type": "timeseries", "title": "Processing Time (P50/P95/P99)", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 7},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "s", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, sum(rate(doc_pipeline_worker_processing_seconds_bucket{job=\"doc-pipeline-worker-ocr\"}[5m])) by (le))", "legendFormat": "P50"},
        {"expr": "histogram_quantile(0.95, sum(rate(doc_pipeline_worker_processing_seconds_bucket{job=\"doc-pipeline-worker-ocr\"}[5m])) by (le))", "legendFormat": "P95"},
        {"expr": "histogram_quantile(0.99, sum(rate(doc_pipeline_worker_processing_seconds_bucket{job=\"doc-pipeline-worker-ocr\"}[5m])) by (le))", "legendFormat": "P99"}
      ]},

    {"type": "timeseries", "title": "Jobs por Status", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 7},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "ops", "custom": {"drawStyle": "line", "fillOpacity": 20}}},
      "options": {"legend": {"calcs": ["mean", "sum"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "rate(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-worker-ocr\",status=\"success\"}[5m])", "legendFormat": "Success"},
        {"expr": "rate(doc_pipeline_jobs_processed_total{job=\"doc-pipeline-worker-ocr\",status=\"error\"}[5m])", "legendFormat": "Error"}
      ]},

    {"type": "timeseries", "title": "Queue Depth", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 15},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short", "custom": {"drawStyle": "line", "fillOpacity": 30}}},
      "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
      "targets": [{"expr": "doc_pipeline_queue_depth{job=\"doc-pipeline-worker-ocr\"}", "legendFormat": "Queue Depth"}]},

    {"type": "timeseries", "title": "Queue Wait Time (P50/P95)", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 15},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "s", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, sum(rate(doc_pipeline_queue_wait_seconds_bucket{job=\"doc-pipeline-worker-ocr\"}[5m])) by (le))", "legendFormat": "P50"},
        {"expr": "histogram_quantile(0.95, sum(rate(doc_pipeline_queue_wait_seconds_bucket{job=\"doc-pipeline-worker-ocr\"}[5m])) by (le))", "legendFormat": "P95"}
      ]}
  ]
}'

echo ""
echo "Concluído!"
echo ""
echo "Dashboards:"
echo "  - Doc Pipeline: $GRAFANA_URL/d/doc-pipeline"
echo "  - Worker OCR:   $GRAFANA_URL/d/worker-ocr"
