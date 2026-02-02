#!/bin/bash
# Auto-scaling para doc-pipeline workers baseado na profundidade da fila
#
# Uso: ./autoscale.sh [--daemon]
#
# Variáveis de ambiente:
#   REDIS_URL          - URL do Redis (default: localhost)
#   MIN_WORKERS        - Mínimo de workers (default: 1)
#   MAX_WORKERS        - Máximo de workers (default: 3)
#   SCALE_UP_THRESHOLD - Queue depth para escalar up (default: 5)
#   SCALE_DOWN_DELAY   - Segundos com fila vazia antes de escalar down (default: 120)
#   CHECK_INTERVAL     - Intervalo entre checagens em segundos (default: 10)
#   METRICS_FILE       - Arquivo para exportar métricas Prometheus (default: /tmp/doc_pipeline_autoscaler.prom)
#
# Exemplos:
#   ./autoscale.sh                    # Roda uma vez
#   ./autoscale.sh --daemon           # Roda continuamente
#   MIN_WORKERS=2 ./autoscale.sh      # Mínimo de 2 workers

set -e

# Configuração
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
MIN_WORKERS="${MIN_WORKERS:-1}"
MAX_WORKERS="${MAX_WORKERS:-3}"
SCALE_UP_THRESHOLD="${SCALE_UP_THRESHOLD:-5}"
SCALE_DOWN_DELAY="${SCALE_DOWN_DELAY:-120}"
CHECK_INTERVAL="${CHECK_INTERVAL:-10}"
QUEUE_NAME="queue:doc:documents"
METRICS_FILE="${METRICS_FILE:-/tmp/doc_pipeline_autoscaler.prom}"

# Estado
EMPTY_SINCE=0
CURRENT_WORKERS=0
SCALE_UP_TOTAL=0
SCALE_DOWN_TOTAL=0

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

export_metrics() {
    local queue_depth=$1
    local workers=$2
    local empty_seconds=$3

    cat > "${METRICS_FILE}.tmp" << METRICS
# HELP doc_pipeline_autoscaler_workers_current Current number of workers
# TYPE doc_pipeline_autoscaler_workers_current gauge
doc_pipeline_autoscaler_workers_current $workers

# HELP doc_pipeline_autoscaler_workers_min Minimum workers configured
# TYPE doc_pipeline_autoscaler_workers_min gauge
doc_pipeline_autoscaler_workers_min $MIN_WORKERS

# HELP doc_pipeline_autoscaler_workers_max Maximum workers configured
# TYPE doc_pipeline_autoscaler_workers_max gauge
doc_pipeline_autoscaler_workers_max $MAX_WORKERS

# HELP doc_pipeline_autoscaler_queue_depth Current queue depth
# TYPE doc_pipeline_autoscaler_queue_depth gauge
doc_pipeline_autoscaler_queue_depth $queue_depth

# HELP doc_pipeline_autoscaler_scale_threshold Queue depth threshold to scale up
# TYPE doc_pipeline_autoscaler_scale_threshold gauge
doc_pipeline_autoscaler_scale_threshold $SCALE_UP_THRESHOLD

# HELP doc_pipeline_autoscaler_scale_up_total Total scale up events
# TYPE doc_pipeline_autoscaler_scale_up_total counter
doc_pipeline_autoscaler_scale_up_total $SCALE_UP_TOTAL

# HELP doc_pipeline_autoscaler_scale_down_total Total scale down events
# TYPE doc_pipeline_autoscaler_scale_down_total counter
doc_pipeline_autoscaler_scale_down_total $SCALE_DOWN_TOTAL

# HELP doc_pipeline_autoscaler_queue_empty_seconds Seconds queue has been empty
# TYPE doc_pipeline_autoscaler_queue_empty_seconds gauge
doc_pipeline_autoscaler_queue_empty_seconds $empty_seconds

# HELP doc_pipeline_autoscaler_last_update_timestamp Last update timestamp
# TYPE doc_pipeline_autoscaler_last_update_timestamp gauge
doc_pipeline_autoscaler_last_update_timestamp $(date +%s)
METRICS
    # Usa cat > em vez de mv para manter o inode (necessário para bind mount do Docker)
    cat "${METRICS_FILE}.tmp" > "$METRICS_FILE"
    rm -f "${METRICS_FILE}.tmp"
}

get_queue_depth() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LLEN "$QUEUE_NAME" 2>/dev/null || echo "0"
}

get_current_workers() {
    docker compose ps --format json 2>/dev/null | grep -c '"Service":"worker"' || echo "0"
}

scale_workers() {
    local target=$1
    local direction=$2  # "up" or "down"
    log "${YELLOW}Scaling workers: $CURRENT_WORKERS → $target${NC}"
    docker compose up -d --scale worker="$target" --no-recreate 2>/dev/null

    # Incrementa contadores
    if [ "$direction" = "up" ]; then
        SCALE_UP_TOTAL=$((SCALE_UP_TOTAL + 1))
    elif [ "$direction" = "down" ]; then
        SCALE_DOWN_TOTAL=$((SCALE_DOWN_TOTAL + 1))
    fi

    CURRENT_WORKERS=$target
}

check_and_scale() {
    local queue_depth=$(get_queue_depth)
    local now=$(date +%s)
    local empty_seconds=0

    # Atualiza contagem de workers
    CURRENT_WORKERS=$(get_current_workers)

    # Log status
    log "Queue: $queue_depth | Workers: $CURRENT_WORKERS/$MAX_WORKERS"

    # Lógica de scale up
    if [ "$queue_depth" -ge "$SCALE_UP_THRESHOLD" ]; then
        EMPTY_SINCE=0

        if [ "$CURRENT_WORKERS" -lt "$MAX_WORKERS" ]; then
            local new_count=$((CURRENT_WORKERS + 1))
            log "${GREEN}↑ Queue depth ($queue_depth) >= threshold ($SCALE_UP_THRESHOLD)${NC}"
            scale_workers "$new_count" "up"
        fi
        export_metrics "$queue_depth" "$CURRENT_WORKERS" 0
        return
    fi

    # Lógica de scale down
    if [ "$queue_depth" -eq 0 ]; then
        if [ "$EMPTY_SINCE" -eq 0 ]; then
            EMPTY_SINCE=$now
            log "Queue empty, starting cooldown ($SCALE_DOWN_DELAY s)"
        fi

        empty_seconds=$((now - EMPTY_SINCE))

        if [ "$empty_seconds" -ge "$SCALE_DOWN_DELAY" ] && [ "$CURRENT_WORKERS" -gt "$MIN_WORKERS" ]; then
            local new_count=$((CURRENT_WORKERS - 1))
            log "${RED}↓ Queue empty for ${empty_seconds}s, scaling down${NC}"
            scale_workers "$new_count" "down"
            EMPTY_SINCE=$now  # Reset timer
            empty_seconds=0
        fi
    else
        EMPTY_SINCE=0
    fi

    # Exporta métricas
    export_metrics "$queue_depth" "$CURRENT_WORKERS" "$empty_seconds"
}

# Main
cd "$(dirname "$0")/.."

log "=== Doc Pipeline Auto-Scaler ==="
log "Config: MIN=$MIN_WORKERS MAX=$MAX_WORKERS THRESHOLD=$SCALE_UP_THRESHOLD DELAY=${SCALE_DOWN_DELAY}s"

# Garante mínimo de workers
CURRENT_WORKERS=$(get_current_workers)
if [ "$CURRENT_WORKERS" -lt "$MIN_WORKERS" ]; then
    log "Starting with minimum workers: $MIN_WORKERS"
    scale_workers "$MIN_WORKERS" "up"
fi

# Exporta métricas iniciais
export_metrics 0 "$CURRENT_WORKERS" 0

if [ "$1" = "--daemon" ]; then
    log "Running in daemon mode (interval: ${CHECK_INTERVAL}s)"
    while true; do
        check_and_scale
        sleep "$CHECK_INTERVAL"
    done
else
    check_and_scale
fi
