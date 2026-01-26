#!/bin/bash
#
# Script para iniciar a API doc-pipeline com ngrok
# Mantém os processos rodando mesmo após sair do SSH
#

set -e

# Configurações
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${DOC_PIPELINE_API_PORT:-8001}"
VENV_PATH="${SCRIPT_DIR}/venv"
LOG_DIR="${SCRIPT_DIR}/logs"
SESSION_API="doc-pipeline-api"
SESSION_NGROK="doc-pipeline-ngrok"

# Ngrok custom domain (para versão paga)
# Pode ser definido via variável de ambiente ou passado como argumento
# Exemplo: NGROK_DOMAIN=meu-dominio.ngrok.io ./start-server.sh
#          ./start-server.sh start meu-dominio.ngrok.io
NGROK_DOMAIN="${NGROK_DOMAIN:-}"

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

    if ! command -v screen &> /dev/null; then
        log_error "screen não está instalado. Instale com: sudo apt install screen"
        exit 1
    fi

    if ! command -v ngrok &> /dev/null; then
        log_error "ngrok não está instalado."
        echo "  Instale com:"
        echo "    curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null"
        echo "    echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list"
        echo "    sudo apt update && sudo apt install ngrok"
        echo "  Ou baixe de: https://ngrok.com/download"
        exit 1
    fi

    if [ ! -d "$VENV_PATH" ]; then
        log_error "Virtual environment não encontrado em: $VENV_PATH"
        exit 1
    fi

    log_info "Todas as dependências OK"
}

# Para sessões existentes
stop_sessions() {
    log_info "Parando sessões existentes..."

    if screen -list | grep -q "$SESSION_API"; then
        screen -S "$SESSION_API" -X quit 2>/dev/null || true
        log_info "Sessão $SESSION_API encerrada"
    fi

    if screen -list | grep -q "$SESSION_NGROK"; then
        screen -S "$SESSION_NGROK" -X quit 2>/dev/null || true
        log_info "Sessão $SESSION_NGROK encerrada"
    fi

    # Aguarda processos encerrarem
    sleep 2
}

# Inicia a API
start_api() {
    log_info "Iniciando API na porta $API_PORT..."

    mkdir -p "$LOG_DIR"

    # Cria script temporário para a API
    cat > /tmp/start_api.sh << EOF
#!/bin/bash
cd "$SCRIPT_DIR"
source "$VENV_PATH/bin/activate"
python api.py 2>&1 | tee "$LOG_DIR/api.log"
EOF
    chmod +x /tmp/start_api.sh

    # Inicia em sessão screen detached
    screen -dmS "$SESSION_API" /tmp/start_api.sh

    log_info "API iniciada em sessão screen: $SESSION_API"

    # Aguarda API subir
    log_info "Aguardando API inicializar..."
    for i in {1..60}; do
        if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            log_info "API pronta!"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    echo ""
    log_error "API não respondeu após 120 segundos. Verifique os logs: $LOG_DIR/api.log"
    exit 1
}

# Inicia ngrok
start_ngrok() {
    log_info "Iniciando ngrok..."

    # Monta comando ngrok
    local ngrok_cmd="ngrok http $API_PORT --log=stdout"

    # Se tem domínio customizado (versão paga), adiciona o parâmetro
    if [ -n "$NGROK_DOMAIN" ]; then
        log_info "Usando domínio customizado: $NGROK_DOMAIN"
        ngrok_cmd="ngrok http $API_PORT --domain=$NGROK_DOMAIN --log=stdout"
    fi

    # Inicia ngrok em sessão screen detached
    screen -dmS "$SESSION_NGROK" bash -c "$ngrok_cmd"

    log_info "ngrok iniciado em sessão screen: $SESSION_NGROK"

    # Aguarda ngrok subir e pega a URL
    log_info "Aguardando ngrok estabelecer túnel..."
    sleep 5

    for i in {1..10}; do
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -oP '"public_url":"https://[^"]+' | head -1 | cut -d'"' -f4)
        if [ -n "$NGROK_URL" ]; then
            break
        fi
        sleep 2
    done

    if [ -z "$NGROK_URL" ]; then
        log_warn "Não foi possível obter a URL do ngrok automaticamente."
        log_warn "Verifique manualmente: http://localhost:4040"
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
    echo "  Local:  http://localhost:$API_PORT"
    if [ -n "$NGROK_URL" ]; then
        echo -e "  ${GREEN}Público: $NGROK_URL${NC}"
    else
        echo "  Público: Verifique em http://localhost:4040"
    fi
    echo ""
    echo "Endpoints:"
    echo "  GET  /health   - Status da API"
    echo "  GET  /classes  - Lista classes suportadas"
    echo "  POST /classify - Classifica documento"
    echo "  POST /extract  - Extrai dados"
    echo "  POST /process  - Pipeline completo"
    echo ""
    echo "Gerenciar sessões screen:"
    echo "  Listar:       screen -ls"
    echo "  Acessar API:  screen -r $SESSION_API"
    echo "  Acessar ngrok: screen -r $SESSION_NGROK"
    echo "  Sair (Ctrl+A, D) para voltar ao terminal sem matar"
    echo ""
    echo "Parar tudo:"
    echo "  $0 stop"
    echo ""
    echo "Logs:"
    echo "  tail -f $LOG_DIR/api.log"
    echo ""
}

# Função para parar
stop() {
    log_info "Parando serviços..."
    stop_sessions
    log_info "Todos os serviços parados"
}

# Função para mostrar status atual
status() {
    echo "Sessões ativas:"
    screen -ls | grep "doc-pipeline" || echo "  Nenhuma sessão doc-pipeline ativa"
    echo ""

    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo -e "API: ${GREEN}ONLINE${NC}"
        curl -s "http://localhost:$API_PORT/health" | python3 -m json.tool 2>/dev/null || true
    else
        echo -e "API: ${RED}OFFLINE${NC}"
    fi
    echo ""

    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -oP '"public_url":"https://[^"]+' | head -1 | cut -d'"' -f4)
    if [ -n "$NGROK_URL" ]; then
        echo -e "ngrok: ${GREEN}ONLINE${NC}"
        echo "  URL: $NGROK_URL"
    else
        echo -e "ngrok: ${RED}OFFLINE${NC}"
    fi
}

# Main
case "${1:-start}" in
    start)
        # Aceita domínio como segundo argumento
        if [ -n "$2" ]; then
            NGROK_DOMAIN="$2"
        fi
        check_dependencies
        stop_sessions
        start_api
        start_ngrok
        show_status
        ;;
    stop)
        stop
        ;;
    restart)
        # Aceita domínio como segundo argumento
        if [ -n "$2" ]; then
            NGROK_DOMAIN="$2"
        fi
        stop
        sleep 2
        check_dependencies
        start_api
        start_ngrok
        show_status
        ;;
    status)
        status
        ;;
    *)
        echo "Uso: $0 {start|stop|restart|status} [ngrok-domain]"
        echo ""
        echo "Exemplos:"
        echo "  $0 start                          # Inicia com URL aleatória do ngrok"
        echo "  $0 start meu-app.ngrok.io         # Inicia com domínio customizado"
        echo "  NGROK_DOMAIN=meu-app.ngrok.io $0  # Via variável de ambiente"
        echo "  $0 stop                           # Para todos os serviços"
        echo "  $0 status                         # Verifica status"
        exit 1
        ;;
esac
