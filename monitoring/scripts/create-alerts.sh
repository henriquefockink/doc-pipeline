#!/bin/bash
# Script para criar alertas no Grafana via API
#
# Uso: ./create-alerts.sh [grafana_url] [api_token] [datasource_uid]
#
# Variáveis de ambiente (ou .env):
#   GRAFANA_URL   - URL do Grafana
#   GRAFANA_TOKEN - Token de API (glsa_xxx) ou user:password
#
# Exemplos:
#   ./create-alerts.sh                                          # usa .env
#   ./create-alerts.sh https://grafana.exemplo.com glsa_xxx     # passa como argumento
#
# Se datasource_uid não for informado, busca automaticamente o primeiro Prometheus.

set -e

# Carrega .env se existir (busca no diretório do script ou no diretório atual)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
elif [ -f ".env" ]; then
    source ".env"
fi

# Parâmetros: argumentos > variáveis de ambiente > defaults
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
        echo "ERRO: Prometheus datasource não encontrado!"
        echo "Datasources disponíveis:"
        echo "$DS_RESPONSE" | grep -o '"name":"[^"]*","type":"[^"]*"' | sed 's/"name":"//g; s/","type":"/ -> /g; s/"//g'
        exit 1
    fi
    echo "$DS_UID"
fi

# Cria folder se nao existe
echo -n "Criando folder 'Doc Pipeline'... "
FOLDER_RESPONSE=$(curl -s -X POST "$GRAFANA_URL/api/folders" \
    -H "$AUTH_HEADER" \
    -H "Content-Type: application/json" \
    -d '{"title": "Doc Pipeline"}' 2>/dev/null || echo '{}')

FOLDER_UID=$(echo "$FOLDER_RESPONSE" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$FOLDER_UID" ]; then
    # Folder ja existe, busca uid
    FOLDER_UID=$(curl -s "$GRAFANA_URL/api/folders" \
        -H "$AUTH_HEADER" | grep -o '"uid":"[^"]*","title":"Doc Pipeline"' | cut -d'"' -f4)
fi

if [ -z "$FOLDER_UID" ]; then
    echo "ERRO: Não foi possível criar/encontrar folder"
    exit 1
fi
echo "$FOLDER_UID"

# Funcao para criar alerta
create_alert() {
    local title="$1"
    local data="$2"
    local condition="$3"
    local for_dur="$4"
    local severity="$5"
    local summary="$6"
    local desc="$7"
    local extra_labels="${8:-}"

    local labels='"severity":"'$severity'","service":"doc-pipeline"'
    [ -n "$extra_labels" ] && labels="$labels,$extra_labels"

    echo -n "  $title... "
    RESP=$(curl -s -w "%{http_code}" -X POST "$GRAFANA_URL/api/v1/provisioning/alert-rules" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -H "X-Disable-Provenance: true" \
        -d '{
            "title": "'"$title"'",
            "ruleGroup": "doc-pipeline-alerts",
            "folderUID": "'"$FOLDER_UID"'",
            "condition": "'"$condition"'",
            "data": '"$data"',
            "noDataState": "OK",
            "execErrState": "Error",
            "for": "'"$for_dur"'",
            "annotations": {"summary":"'"$summary"'","description":"'"$desc"'"},
            "labels": {'"$labels"'}
        }')

    CODE="${RESP: -3}"
    BODY="${RESP:0:-3}"

    if [ "$CODE" = "201" ]; then
        echo "OK"
    elif echo "$BODY" | grep -q "already exists"; then
        echo "SKIP (exists)"
    else
        echo "FAIL ($CODE)"
    fi
}

echo ""
echo "Criando alertas..."

# 1. High Error Rate (5xx)
create_alert "High Error Rate (5xx)" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_requests_total{status=~\"5..\",endpoint=~\"/classify|/extract|/process\"}[5m]))"}},
    {"refId":"B","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=~\"/classify|/extract|/process\"}[5m]))"}},
    {"refId":"C","datasourceUid":"__expr__","model":{"type":"math","expression":"$A / $B"}},
    {"refId":"D","datasourceUid":"__expr__","model":{"type":"threshold","expression":"C","conditions":[{"evaluator":{"type":"gt","params":[0.05]}}]}}
]' "D" "5m" "critical" "High 5xx error rate" "Error rate above 5%"

# 2. High Latency P95
create_alert "High Latency P95" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"histogram_quantile(0.95,sum(rate(doc_pipeline_request_duration_seconds_bucket{endpoint=~\"/classify|/extract|/process\"}[5m])) by (le))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[30]}}]}}
]' "B" "5m" "warning" "High P95 latency" "P95 latency above 30s"

# 3. Critical Latency P99
create_alert "Critical Latency P99" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"histogram_quantile(0.99,sum(rate(doc_pipeline_request_duration_seconds_bucket{endpoint=~\"/classify|/extract|/process\"}[5m])) by (le))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[60]}}]}}
]' "B" "5m" "critical" "Critical P99 latency" "P99 latency above 60s"

# 4. High Concurrency
create_alert "High Concurrency" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(doc_pipeline_requests_in_progress{endpoint=~\"/classify|/extract|/process\"})"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[10]}}]}}
]' "B" "5m" "warning" "High concurrency" "More than 10 concurrent requests"

# 5. Low Classification Confidence
create_alert "Low Classification Confidence" '[
    {"refId":"A","relativeTimeRange":{"from":600,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"histogram_quantile(0.5,sum(rate(doc_pipeline_classification_confidence_bucket[10m])) by (le))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"lt","params":[0.7]}}]}}
]' "B" "15m" "warning" "Low classification confidence" "Median confidence below 0.7"

# 6. Very Low Confidence
create_alert "Very Low Confidence" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"histogram_quantile(0.5,sum(rate(doc_pipeline_classification_confidence_bucket[5m])) by (le))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"lt","params":[0.5]}}]}}
]' "B" "5m" "critical" "Very low confidence" "Median confidence below 0.5 - model degraded"

# 7. /classify Errors
create_alert "/classify Errors" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/classify\",status=~\"5..\"}[5m])) / sum(rate(doc_pipeline_requests_total{endpoint=\"/classify\"}[5m]))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[0.05]}}]}}
]' "B" "5m" "critical" "High error rate on /classify" "Classification error rate above 5%" '"endpoint":"classify"'

# 8. /extract Errors
create_alert "/extract Errors" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/extract\",status=~\"5..\"}[5m])) / sum(rate(doc_pipeline_requests_total{endpoint=\"/extract\"}[5m]))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[0.05]}}]}}
]' "B" "5m" "critical" "High error rate on /extract" "Extraction error rate above 5%" '"endpoint":"extract"'

# 9. /process Errors
create_alert "/process Errors" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_requests_total{endpoint=\"/process\",status=~\"5..\"}[5m])) / sum(rate(doc_pipeline_requests_total{endpoint=\"/process\"}[5m]))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[0.05]}}]}}
]' "B" "5m" "critical" "High error rate on /process" "Full pipeline error rate above 5%" '"endpoint":"process"'

echo ""
echo "Criando alertas de Queue & Worker..."

# 10. High Queue Depth
create_alert "High Queue Depth" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"doc_pipeline_queue_depth"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[50]}}]}}
]' "B" "5m" "warning" "Queue depth is high" "Jobs waiting in queue above 50" '"component":"queue"'

# 11. Critical Queue Depth
create_alert "Critical Queue Depth" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"doc_pipeline_queue_depth"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[90]}}]}}
]' "B" "2m" "critical" "Queue depth critical" "Queue near capacity (>90 jobs)" '"component":"queue"'

# 12. High Queue Wait Time
create_alert "High Queue Wait Time" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"histogram_quantile(0.95,sum(rate(doc_pipeline_queue_wait_seconds_bucket[5m])) by (le))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[60]}}]}}
]' "B" "5m" "warning" "Jobs waiting too long" "P95 queue wait time above 60s" '"component":"queue"'

# 13. Worker Not Processing
create_alert "Worker Not Processing" '[
    {"refId":"A","relativeTimeRange":{"from":600,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"doc_pipeline_queue_depth"}},
    {"refId":"B","relativeTimeRange":{"from":600,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total[5m]))"}},
    {"refId":"C","datasourceUid":"__expr__","model":{"type":"math","expression":"$A > 0 && $B == 0"}},
    {"refId":"D","datasourceUid":"__expr__","model":{"type":"threshold","expression":"C","conditions":[{"evaluator":{"type":"gt","params":[0]}}]}}
]' "D" "5m" "critical" "Worker not processing jobs" "Queue has jobs but worker is not processing" '"component":"worker"'

# 14. Worker Error Rate
create_alert "Worker Error Rate" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total{status=\"error\"}[5m])) / sum(rate(doc_pipeline_jobs_processed_total[5m]))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[0.1]}}]}}
]' "B" "5m" "critical" "High worker error rate" "More than 10% of jobs failing" '"component":"worker"'

# 15. Webhook Failures
create_alert "Webhook Failures" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_webhook_deliveries_total{status=\"failed\"}[5m]))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[0.1]}}]}}
]' "B" "5m" "warning" "Webhook deliveries failing" "Webhooks failing to deliver" '"component":"webhook"'

echo ""
echo "Criando alertas do OCR Worker..."

# 16. OCR Worker Down
create_alert "OCR Worker Down" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"up{job=\"doc-pipeline-worker-ocr\"}"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"lt","params":[1]}}]}}
]' "B" "2m" "critical" "OCR Worker is down" "The OCR worker has been unreachable for more than 2 minutes" '"worker":"ocr"'

# 17. OCR High Error Rate
create_alert "OCR High Error Rate" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"sum(rate(doc_pipeline_jobs_processed_total{operation=\"ocr\",status=\"error\"}[5m])) / sum(rate(doc_pipeline_jobs_processed_total{operation=\"ocr\"}[5m]))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[0.1]}}]}}
]' "B" "5m" "warning" "High OCR error rate" "OCR error rate above 10%" '"worker":"ocr"'

# 18. OCR High Latency
create_alert "OCR High Latency" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"histogram_quantile(0.95,rate(doc_pipeline_worker_processing_seconds_bucket{operation=\"ocr\"}[5m]))"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[30]}}]}}
]' "B" "5m" "warning" "OCR processing time is high" "P95 processing time above 30 seconds" '"worker":"ocr"'

# 19. OCR Queue Backup
create_alert "OCR Queue Backup" '[
    {"refId":"A","relativeTimeRange":{"from":300,"to":0},"datasourceUid":"'$DS_UID'","model":{"expr":"doc_pipeline_queue_depth"}},
    {"refId":"B","datasourceUid":"__expr__","model":{"type":"threshold","expression":"A","conditions":[{"evaluator":{"type":"gt","params":[10]}}]}}
]' "B" "5m" "warning" "OCR queue is backing up" "Queue depth above 10 jobs" '"worker":"ocr","component":"queue"'

echo ""
echo "Concluído! Verifique em: $GRAFANA_URL/alerting/list"
