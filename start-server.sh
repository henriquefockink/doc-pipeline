#!/bin/bash
#
# Script para iniciar a API doc-pipeline com ngrok
# Mantém os processos rodando mesmo após sair do SSH (usa nohup)
#

set -e

# Configurações
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${DOC_PIPELINE_API_PORT:-9000}"
VENV_PATH="${SCRIPT_DIR}/venv"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_DIR="${SCRIPT_DIR}/pids"

# Ngrok
NGROK_DOMAIN="${NGROK_DOMAIN:-}"
NGROK_ENABLED=false

# Foreground mode (não solta o terminal)
FOREGROUND=false

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verifica dependências
check_dependencies() {
    log_info "Verificando dependências..."

    if ! command -v ngrok &> /dev/null; then
        log_warn "ngrok não está instalado. Túnel externo não estará disponível."
        log_warn "Instale com: https://ngrok.com/download"
    fi

    if [ ! -d "$VENV_PATH" ]; then
        log_error "Virtual environment não encontrado em: $VENV_PATH"
        exit 1
    fi

    log_info "Dependências OK"
}

# Para processos existentes
stop_processes() {
    log_info "Parando processos existentes..."

    # Para API
    if [ -f "$PID_DIR/api.pid" ]; then
        pid=$(cat "$PID_DIR/api.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            log_info "API (PID $pid) encerrada"
        fi
        rm -f "$PID_DIR/api.pid"
    fi

    # Para ngrok
    if [ -f "$PID_DIR/ngrok.pid" ]; then
        pid=$(cat "$PID_DIR/ngrok.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            log_info "ngrok (PID $pid) encerrado"
        fi
        rm -f "$PID_DIR/ngrok.pid"
    fi

    # Mata processos órfãos
    pkill -f "python api.py" 2>/dev/null || true
    pkill -f "ngrok http" 2>/dev/null || true

    sleep 2
}

# Inicia a API
start_api() {
    log_info "Iniciando API na porta $API_PORT..."

    mkdir -p "$LOG_DIR" "$PID_DIR"

    cd "$SCRIPT_DIR"
    source "$VENV_PATH/bin/activate"

    # Modo foreground: roda direto no terminal
    if [ "$FOREGROUND" = true ]; then
        log_info "Modo foreground - logs no terminal (Ctrl+C para parar)"
        echo ""
        exec python api.py
    fi

    # Modo background: usa nohup
    > "$LOG_DIR/api.log"  # Limpa log anterior
    nohup python api.py >> "$LOG_DIR/api.log" 2>&1 &
    echo $! > "$PID_DIR/api.pid"

    log_info "API iniciada (PID $(cat $PID_DIR/api.pid))"

    # Aguarda API subir mostrando progresso do warmup
    log_info "Aguardando warmup dos modelos..."

    local last_component=""
    for i in {1..180}; do  # 6 minutos (3 modelos podem demorar)
        # Verifica se API está pronta
        if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            echo ""
            log_info "API pronta!"
            return 0
        fi

        # Verifica se processo ainda está rodando
        if ! kill -0 "$(cat $PID_DIR/api.pid 2>/dev/null)" 2>/dev/null; then
            echo ""
            log_error "API encerrou inesperadamente. Verifique: $LOG_DIR/api.log"
            tail -20 "$LOG_DIR/api.log"
            exit 1
        fi

        # Mostra eventos de warmup do log
        if [ -f "$LOG_DIR/api.log" ]; then
            # Procura por warmup_start e warmup_complete
            while IFS= read -r line; do
                if echo "$line" | grep -q '"event":"warmup_start"'; then
                    component=$(echo "$line" | grep -oP '"component":"[^"]+' | cut -d'"' -f4)
                    if [ -n "$component" ] && [ "$component" != "$last_component" ]; then
                        echo -e "\n  ${YELLOW}⏳ Carregando: $component${NC}"
                        last_component="$component"
                    fi
                elif echo "$line" | grep -q '"event":"warmup_complete"'; then
                    component=$(echo "$line" | grep -oP '"component":"[^"]+' | cut -d'"' -f4)
                    if [ -n "$component" ]; then
                        echo -e "  ${GREEN}✓ $component${NC}"
                    fi
                fi
            done < <(tail -50 "$LOG_DIR/api.log" 2>/dev/null)
        fi

        sleep 2
    done
    echo ""
    log_error "API não respondeu após 6 minutos. Verifique: $LOG_DIR/api.log"
    exit 1
}

# Inicia ngrok
start_ngrok() {
    if ! command -v ngrok &> /dev/null; then
        log_warn "ngrok não instalado, pulando túnel externo"
        return 0
    fi

    log_info "Iniciando ngrok..."

    mkdir -p "$PID_DIR"

    # Monta comando ngrok
    local ngrok_cmd="ngrok http $API_PORT --log=stdout"

    if [ -n "$NGROK_DOMAIN" ]; then
        log_info "Usando domínio customizado: $NGROK_DOMAIN"
        ngrok_cmd="ngrok http $API_PORT --domain=$NGROK_DOMAIN --log=stdout"
    fi

    # Inicia com nohup
    nohup $ngrok_cmd > "$LOG_DIR/ngrok.log" 2>&1 &
    echo $! > "$PID_DIR/ngrok.pid"

    log_info "ngrok iniciado (PID $(cat $PID_DIR/ngrok.pid))"

    # Aguarda ngrok subir e pega a URL
    log_info "Aguardando túnel..."
    sleep 5

    for i in {1..10}; do
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -oP '"public_url":"https://[^"]+' | head -1 | cut -d'"' -f4)
        if [ -n "$NGROK_URL" ]; then
            break
        fi
        sleep 2
    done

    if [ -z "$NGROK_URL" ]; then
        log_warn "Não foi possível obter URL do ngrok. Verifique: http://localhost:4040"
    fi
}

# Mostra status
show_status() {
    echo ""
    echo "============================================"
    echo -e "${GREEN}Servidor iniciado com sucesso!${NC}"
    echo "============================================"
    echo ""
    echo "URLs:"
    echo "  Local: http://localhost:$API_PORT"
    if [ "$NGROK_ENABLED" = true ]; then
        if [ -n "$NGROK_URL" ]; then
            echo -e "  ${GREEN}Público: $NGROK_URL${NC}"
        else
            echo "  Público: Verifique em http://localhost:4040"
        fi
    fi
    echo ""
    echo "Endpoints:"
    echo "  GET  /health              - Status da API"
    echo "  GET  /classes             - Lista classes suportadas"
    echo "  POST /classify            - Classifica documento"
    echo "  POST /extract?doc_type=X  - Extrai dados (generic para OCR/PDF)"
    echo "  POST /process             - Pipeline completo"
    echo ""
    echo "Gerenciar:"
    echo "  $0 status   # Ver status"
    echo "  $0 stop     # Parar tudo"
    echo "  $0 restart  # Reiniciar"
    echo ""
    echo "Logs:"
    echo "  tail -f $LOG_DIR/api.log"
    echo ""
}

# Função para parar
stop() {
    stop_processes
    log_info "Todos os serviços parados"
}

# Função para mostrar status atual
status() {
    echo "Processos:"

    if [ -f "$PID_DIR/api.pid" ]; then
        pid=$(cat "$PID_DIR/api.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  API:   ${GREEN}ONLINE${NC} (PID $pid)"
        else
            echo -e "  API:   ${RED}OFFLINE${NC} (PID file existe mas processo morto)"
        fi
    else
        echo -e "  API:   ${RED}OFFLINE${NC}"
    fi

    if [ -f "$PID_DIR/ngrok.pid" ]; then
        pid=$(cat "$PID_DIR/ngrok.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ngrok: ${GREEN}ONLINE${NC} (PID $pid)"
        else
            echo -e "  ngrok: ${RED}OFFLINE${NC}"
        fi
    else
        echo -e "  ngrok: ${RED}OFFLINE${NC}"
    fi

    echo ""

    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo -e "Health: ${GREEN}OK${NC}"
        curl -s "http://localhost:$API_PORT/health" | python3 -m json.tool 2>/dev/null || true
    else
        echo -e "Health: ${RED}FALHOU${NC}"
    fi

    echo ""

    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -oP '"public_url":"https://[^"]+' | head -1 | cut -d'"' -f4)
    if [ -n "$NGROK_URL" ]; then
        echo -e "URL pública: ${GREEN}$NGROK_URL${NC}"
    fi
}

# Processa argumentos
parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --ngrok)
                NGROK_ENABLED=true
                ;;
            --ngrok=*)
                NGROK_ENABLED=true
                NGROK_DOMAIN="${arg#*=}"
                ;;
            --foreground|-f)
                FOREGROUND=true
                ;;
            -*)
                # ignora outras flags
                ;;
            *)
                # Se parece com domínio (contém .), assume que é ngrok domain
                if [[ "$arg" == *.* ]]; then
                    NGROK_ENABLED=true
                    NGROK_DOMAIN="$arg"
                fi
                ;;
        esac
    done
}

# Main
case "${1:-start}" in
    start)
        shift || true
        parse_args "$@"
        check_dependencies
        stop_processes
        start_api
        if [ "$NGROK_ENABLED" = true ]; then
            start_ngrok
        fi
        if [ "$FOREGROUND" = false ]; then
            show_status
        fi
        ;;
    stop)
        stop
        ;;
    restart)
        shift || true
        parse_args "$@"
        stop
        sleep 2
        check_dependencies
        start_api
        if [ "$NGROK_ENABLED" = true ]; then
            start_ngrok
        fi
        if [ "$FOREGROUND" = false ]; then
            show_status
        fi
        ;;
    status)
        status
        ;;
    *)
        echo "Uso: $0 {start|stop|restart|status} [opções]"
        echo ""
        echo "Opções:"
        echo "  -f, --foreground        Roda em foreground (logs no terminal)"
        echo "  --ngrok                 Habilita túnel ngrok (URL aleatória)"
        echo "  --ngrok=DOMINIO         Habilita ngrok com domínio fixo"
        echo ""
        echo "Exemplos:"
        echo "  $0 start                        # API em background"
        echo "  $0 start -f                     # API em foreground (ver logs)"
        echo "  $0 start --ngrok                # API + ngrok"
        echo "  $0 stop                         # Para tudo"
        echo "  $0 status                       # Verifica status"
        exit 1
        ;;
esac
