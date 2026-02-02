#!/bin/bash
# Script para criar dashboards no Grafana via API
#
# Uso: ./create-dashboards.sh [grafana_url] [api_token]
#
# Variáveis de ambiente (ou .env):
#   GRAFANA_URL   - URL do Grafana
#   GRAFANA_TOKEN - Token de API (glsa_xxx) ou user:password
#
# Exemplos:
#   ./create-dashboards.sh                                          # usa .env
#   ./create-dashboards.sh https://grafana.exemplo.com glsa_xxx     # passa como argumento

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

# Cria folder "Doc Pipeline" se não existe
create_folder() {
    local title="$1"
    local uid="$2"

    echo -n "Criando folder '$title'... "
    RESP=$(curl -s -X POST "$GRAFANA_URL/api/folders" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -d '{"title": "'"$title"'", "uid": "'"$uid"'"}' 2>/dev/null || echo '{}')

    FOLDER_UID=$(echo "$RESP" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ -z "$FOLDER_UID" ]; then
        # Folder já existe, busca uid
        FOLDER_UID=$(curl -s "$GRAFANA_URL/api/folders/$uid" -H "$AUTH_HEADER" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ -z "$FOLDER_UID" ]; then
        echo "ERRO"
        return 1
    fi
    echo "$FOLDER_UID"
    echo "$FOLDER_UID"
}

# Cria folders
FOLDER_MAIN=$(create_folder "Doc Pipeline" "doc-pipeline")
FOLDER_WORKERS=$(create_folder "Doc Pipeline - Workers" "doc-pipeline-workers")

# Função para criar/atualizar dashboard
create_dashboard() {
    local title="$1"
    local uid="$2"
    local folder_uid="$3"
    local dashboard_json="$4"

    echo -n "  $title... "

    # Substitui datasource UID no JSON
    dashboard_json=$(echo "$dashboard_json" | sed "s/\${datasource}/$DS_UID/g; s/\"uid\": \"prometheus\"/\"uid\": \"$DS_UID\"/g")

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
    fi
}

echo ""
echo "Criando dashboards..."

# ============================================================
# Dashboard: Worker OCR
# ============================================================
create_dashboard "Worker OCR" "worker-ocr" "$FOLDER_WORKERS" '{
  "uid": "worker-ocr",
  "title": "Worker OCR",
  "tags": ["doc-pipeline", "ocr", "worker"],
  "timezone": "browser",
  "refresh": "30s",
  "time": {"from": "now-1h", "to": "now"},
  "panels": [
    {
      "type": "row",
      "title": "Overview",
      "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}
    },
    {
      "type": "stat",
      "title": "Queue Depth",
      "gridPos": {"h": 6, "w": 4, "x": 0, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 5}, {"color": "red", "value": 10}]},
          "unit": "none"
        }
      },
      "options": {"colorMode": "value", "graphMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "doc_pipeline_queue_depth", "legendFormat": "Queue"}]
    },
    {
      "type": "stat",
      "title": "Jobs/s",
      "gridPos": {"h": 6, "w": 4, "x": 4, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "ops"}},
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(rate(doc_pipeline_jobs_processed_total{operation=\"ocr\"}[5m]))", "legendFormat": "Jobs/s"}]
    },
    {
      "type": "stat",
      "title": "P95 Latency",
      "gridPos": {"h": 6, "w": 4, "x": 8, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 5}, {"color": "red", "value": 15}]},
          "unit": "s"
        }
      },
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "histogram_quantile(0.95, rate(doc_pipeline_worker_processing_seconds_bucket{operation=\"ocr\"}[5m]))", "legendFormat": "P95"}]
    },
    {
      "type": "stat",
      "title": "Error Rate",
      "gridPos": {"h": 6, "w": 4, "x": 12, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 0.01}, {"color": "red", "value": 0.05}]},
          "unit": "percentunit"
        }
      },
      "options": {"colorMode": "value", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(rate(doc_pipeline_jobs_processed_total{operation=\"ocr\",status=\"error\"}[5m])) / sum(rate(doc_pipeline_jobs_processed_total{operation=\"ocr\"}[5m]))", "legendFormat": "Error Rate"}]
    },
    {
      "type": "stat",
      "title": "Jobs (24h)",
      "gridPos": {"h": 6, "w": 4, "x": 16, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short"}},
      "options": {"colorMode": "value", "graphMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "sum(increase(doc_pipeline_jobs_processed_total{operation=\"ocr\",status=\"success\"}[24h]))", "legendFormat": "Jobs"}]
    },
    {
      "type": "stat",
      "title": "Worker Status",
      "gridPos": {"h": 6, "w": 4, "x": 20, "y": 1},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "mappings": [{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}, "1": {"text": "UP", "color": "green"}}}],
          "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": null}, {"color": "green", "value": 1}]}
        }
      },
      "options": {"colorMode": "background", "graphMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"expr": "up{job=\"doc-pipeline-worker-ocr\"}", "legendFormat": "Status"}]
    },
    {
      "type": "row",
      "title": "Processing Metrics",
      "gridPos": {"h": 1, "w": 24, "x": 0, "y": 7}
    },
    {
      "type": "timeseries",
      "title": "Processing Time (P50/P95/P99)",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "s", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, rate(doc_pipeline_worker_processing_seconds_bucket{operation=\"ocr\"}[5m]))", "legendFormat": "P50"},
        {"expr": "histogram_quantile(0.95, rate(doc_pipeline_worker_processing_seconds_bucket{operation=\"ocr\"}[5m]))", "legendFormat": "P95"},
        {"expr": "histogram_quantile(0.99, rate(doc_pipeline_worker_processing_seconds_bucket{operation=\"ocr\"}[5m]))", "legendFormat": "P99"}
      ]
    },
    {
      "type": "timeseries",
      "title": "Jobs por Status",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "ops", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "sum"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "rate(doc_pipeline_jobs_processed_total{operation=\"ocr\",status=\"success\"}[5m])", "legendFormat": "Success"},
        {"expr": "rate(doc_pipeline_jobs_processed_total{operation=\"ocr\",status=\"error\"}[5m])", "legendFormat": "Error"}
      ]
    },
    {
      "type": "timeseries",
      "title": "Queue Depth",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short", "custom": {"drawStyle": "line", "fillOpacity": 30}}},
      "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
      "targets": [{"expr": "doc_pipeline_queue_depth", "legendFormat": "Queue Depth"}]
    },
    {
      "type": "timeseries",
      "title": "Queue Wait Time (P50/P95)",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "s", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "bottom"}},
      "targets": [
        {"expr": "histogram_quantile(0.50, rate(doc_pipeline_queue_wait_seconds_bucket[5m]))", "legendFormat": "P50"},
        {"expr": "histogram_quantile(0.95, rate(doc_pipeline_queue_wait_seconds_bucket[5m]))", "legendFormat": "P95"}
      ]
    },
    {
      "type": "row",
      "title": "Delivery",
      "gridPos": {"h": 1, "w": 24, "x": 0, "y": 24}
    },
    {
      "type": "timeseries",
      "title": "Jobs por Delivery Mode",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 25},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short", "custom": {"drawStyle": "bars", "fillOpacity": 100, "stacking": {"mode": "normal"}}}},
      "options": {"legend": {"calcs": ["sum"], "displayMode": "table", "placement": "bottom"}},
      "targets": [{"expr": "increase(doc_pipeline_jobs_processed_total{operation=\"ocr\"}[1h])", "legendFormat": "{{delivery_mode}}"}]
    },
    {
      "type": "timeseries",
      "title": "Webhook Deliveries",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 25},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "ops", "custom": {"drawStyle": "line", "fillOpacity": 10}}},
      "options": {"legend": {"calcs": ["mean", "sum"], "displayMode": "table", "placement": "bottom"}},
      "targets": [{"expr": "rate(doc_pipeline_webhook_deliveries_total[5m])", "legendFormat": "{{status}}"}]
    }
  ]
}'

echo ""
echo "Concluído! Verifique em: $GRAFANA_URL/dashboards"
