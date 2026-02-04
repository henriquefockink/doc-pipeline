#!/bin/bash
# Auto-scaling para doc-pipeline workers baseado na profundidade da fila
#
# Uso: ./autoscale.sh [--daemon]
#
# Variáveis de ambiente:
#   REDIS_HOST         - Host do Redis (default: localhost)
#   REDIS_PORT         - Porta do Redis (default: 6379)
#   MIN_WORKERS        - Mínimo de workers (default: 1)
#   MAX_WORKERS        - Máximo de workers (default: 3)
#   SCALE_UP_THRESHOLD - Queue depth para escalar up (default: 5)
#   SCALE_DOWN_DELAY   - Segundos com fila vazia antes de escalar down (default: 120)
#   CHECK_INTERVAL     - Intervalo entre checagens em segundos (default: 10)
#   METRICS_FILE       - Arquivo para exportar métricas Prometheus
#
# Workers pré-definidos:
#   - worker-docid-1 (porta 9010) - sempre ativo
#   - worker-docid-2 (porta 9012) - sob demanda
#   - worker-docid-3 (porta 9014) - sob demanda

# NOTE: Não usar "set -e" pois ((count++)) retorna exit code 1 quando count=0
# e isso mata o script silenciosamente

# Trap para logar erros inesperados
trap 'log "ERROR: Script died unexpectedly at line $LINENO"' ERR

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

# Workers disponíveis (em ordem)
WORKERS=("worker-docid-1" "worker-docid-2" "worker-docid-3")

# Estado
EMPTY_SINCE=0
SCALE_UP_TOTAL=0
SCALE_DOWN_TOTAL=0

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

get_queue_depth() {
    timeout 5 redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LLEN "$QUEUE_NAME" 2>/dev/null || echo "0"
}

# Verifica se há warmup ativo no Redis
# Retorna: "workers:N" se ativo, "inactive" se não
get_warmup_status() {
    local warmup_data
    warmup_data=$(timeout 5 redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" GET "autoscaler:warmup" 2>/dev/null)

    if [ -z "$warmup_data" ] || [ "$warmup_data" = "(nil)" ]; then
        echo "inactive"
        return
    fi

    # Parse JSON to get workers count (handles optional spaces)
    local workers
    workers=$(echo "$warmup_data" | grep -oE '"workers":\s*[0-9]+' | grep -oE '[0-9]+')

    if [ -n "$workers" ]; then
        echo "workers:$workers"
    else
        echo "inactive"
    fi
}

# Retorna lista de workers rodando
get_running_workers() {
    local running=""
    for worker in "${WORKERS[@]}"; do
        if timeout 10 docker compose ps --status running --format '{{.Service}}' 2>/dev/null | grep -q "^${worker}$"; then
            running="$running $worker"
        fi
    done
    echo "$running"
}

# Conta workers rodando
count_running_workers() {
    local count=0
    for worker in "${WORKERS[@]}"; do
        if timeout 10 docker compose ps --status running --format '{{.Service}}' 2>/dev/null | grep -q "^${worker}$"; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

# Inicia o próximo worker disponível
start_next_worker() {
    for worker in "${WORKERS[@]}"; do
        if ! timeout 10 docker compose ps --status running --format '{{.Service}}' 2>/dev/null | grep -q "^${worker}$"; then
            log "${GREEN}↑ Starting $worker${NC}"
            timeout 60 docker compose up -d "$worker" 2>/dev/null || log "${YELLOW}⚠ Timeout starting $worker${NC}"
            SCALE_UP_TOTAL=$((SCALE_UP_TOTAL + 1))
            return 0
        fi
    done
    return 1
}

# Para o último worker ativo (exceto o primeiro)
stop_last_worker() {
    # Itera de trás pra frente, nunca para o primeiro
    for ((i=${#WORKERS[@]}-1; i>0; i--)); do
        local worker="${WORKERS[$i]}"
        if timeout 10 docker compose ps --status running --format '{{.Service}}' 2>/dev/null | grep -q "^${worker}$"; then
            log "${RED}↓ Stopping $worker${NC}"
            timeout 60 docker compose stop "$worker" 2>/dev/null || log "${YELLOW}⚠ Timeout stopping $worker${NC}"
            SCALE_DOWN_TOTAL=$((SCALE_DOWN_TOTAL + 1))
            return 0
        fi
    done
    return 1
}

export_metrics() {
    local queue_depth=$1
    local workers=$2
    local empty_seconds=$3
    local cooldown_remaining=0

    if [ "$empty_seconds" -gt 0 ] && [ "$empty_seconds" -lt "$SCALE_DOWN_DELAY" ]; then
        cooldown_remaining=$((SCALE_DOWN_DELAY - empty_seconds))
    fi

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

# HELP doc_pipeline_autoscaler_cooldown_remaining Seconds until scale down allowed
# TYPE doc_pipeline_autoscaler_cooldown_remaining gauge
doc_pipeline_autoscaler_cooldown_remaining $cooldown_remaining

# HELP doc_pipeline_autoscaler_last_update_timestamp Last update timestamp
# TYPE doc_pipeline_autoscaler_last_update_timestamp gauge
doc_pipeline_autoscaler_last_update_timestamp $(date +%s)
METRICS
    cat "${METRICS_FILE}.tmp" > "$METRICS_FILE" 2>/dev/null || true
    rm -f "${METRICS_FILE}.tmp" 2>/dev/null || true
}

check_and_scale() {
    local queue_depth=$(get_queue_depth)
    local current_workers=$(count_running_workers)
    local now=$(date +%s)
    local empty_seconds=0

    # Check warmup status
    local warmup_status=$(get_warmup_status)
    local warmup_workers=0
    local warmup_active=false

    if [[ "$warmup_status" == workers:* ]]; then
        warmup_workers=${warmup_status#workers:}
        warmup_active=true
    fi

    # Log status
    local running=$(get_running_workers)
    if [ "$warmup_active" = true ]; then
        log "Queue: ${BLUE}$queue_depth${NC} | Workers: ${BLUE}$current_workers/$MAX_WORKERS${NC} | ${YELLOW}WARMUP($warmup_workers)${NC} |$running"
    else
        log "Queue: ${BLUE}$queue_depth${NC} | Workers: ${BLUE}$current_workers/$MAX_WORKERS${NC} |$running"
    fi

    # Warmup mode: scale up to requested workers
    if [ "$warmup_active" = true ] && [ "$current_workers" -lt "$warmup_workers" ]; then
        log "${GREEN}↑ Warmup active, scaling to $warmup_workers workers${NC}"
        while [ "$current_workers" -lt "$warmup_workers" ]; do
            start_next_worker
            current_workers=$((current_workers + 1))
        done
        EMPTY_SINCE=0
        export_metrics "$queue_depth" "$current_workers" 0
        return
    fi

    # Lógica de scale up (queue-based)
    if [ "$queue_depth" -ge "$SCALE_UP_THRESHOLD" ]; then
        EMPTY_SINCE=0

        if [ "$current_workers" -lt "$MAX_WORKERS" ]; then
            log "${GREEN}↑ Queue depth ($queue_depth) >= threshold ($SCALE_UP_THRESHOLD)${NC}"
            start_next_worker
            current_workers=$((current_workers + 1))
        fi
        export_metrics "$queue_depth" "$current_workers" 0
        return
    fi

    # Lógica de scale down
    if [ "$queue_depth" -eq 0 ]; then
        # Determine minimum workers (normal or warmup)
        local effective_min=$MIN_WORKERS
        if [ "$warmup_active" = true ] && [ "$warmup_workers" -gt "$MIN_WORKERS" ]; then
            effective_min=$warmup_workers
        fi

        if [ "$EMPTY_SINCE" -eq 0 ]; then
            EMPTY_SINCE=$now
            if [ "$warmup_active" = true ]; then
                log "Queue empty, warmup active (min $warmup_workers workers)"
            else
                log "Queue empty, starting cooldown (${SCALE_DOWN_DELAY}s)"
            fi
        fi

        empty_seconds=$((now - EMPTY_SINCE))

        # Only scale down if: cooldown passed AND not in warmup mode (or below warmup target)
        if [ "$empty_seconds" -ge "$SCALE_DOWN_DELAY" ] && [ "$current_workers" -gt "$effective_min" ]; then
            log "${RED}↓ Queue empty for ${empty_seconds}s, scaling down (min: $effective_min)${NC}"
            stop_last_worker
            current_workers=$((current_workers - 1))
            EMPTY_SINCE=$now  # Reset timer
            empty_seconds=0
        fi
    else
        EMPTY_SINCE=0
    fi

    export_metrics "$queue_depth" "$current_workers" "$empty_seconds"
}

# Main
cd "$(dirname "$0")/.."

log "=== Doc Pipeline Auto-Scaler ==="
log "Config: MIN=$MIN_WORKERS MAX=$MAX_WORKERS THRESHOLD=$SCALE_UP_THRESHOLD DELAY=${SCALE_DOWN_DELAY}s"
log "Workers: ${WORKERS[*]}"

# Garante que pelo menos MIN_WORKERS estejam rodando
current=$(count_running_workers)
if [ "$current" -lt "$MIN_WORKERS" ]; then
    log "Starting minimum workers ($MIN_WORKERS)..."
    for ((i=0; i<MIN_WORKERS; i++)); do
        worker="${WORKERS[$i]}"
        if ! timeout 10 docker compose ps --status running --format '{{.Service}}' 2>/dev/null | grep -q "^${worker}$"; then
            timeout 60 docker compose up -d "$worker" 2>/dev/null
        fi
    done
fi

# Exporta métricas iniciais
export_metrics 0 "$(count_running_workers)" 0

if [ "$1" = "--daemon" ]; then
    log "Running in daemon mode (interval: ${CHECK_INTERVAL}s)"
    while true; do
        check_and_scale
        sleep "$CHECK_INTERVAL"
    done
else
    check_and_scale
fi
