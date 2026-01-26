#!/bin/bash
#
# Script para parar a API doc-pipeline e ngrok
#

SESSION_API="doc-pipeline-api"
SESSION_NGROK="doc-pipeline-ngrok"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Parando serviços doc-pipeline..."

# Para sessão da API
if screen -list | grep -q "$SESSION_API"; then
    screen -S "$SESSION_API" -X quit 2>/dev/null
    echo -e "${GREEN}[OK]${NC} Sessão $SESSION_API encerrada"
else
    echo -e "${RED}[--]${NC} Sessão $SESSION_API não encontrada"
fi

# Para sessão do ngrok
if screen -list | grep -q "$SESSION_NGROK"; then
    screen -S "$SESSION_NGROK" -X quit 2>/dev/null
    echo -e "${GREEN}[OK]${NC} Sessão $SESSION_NGROK encerrada"
else
    echo -e "${RED}[--]${NC} Sessão $SESSION_NGROK não encontrada"
fi

echo ""
echo "Serviços parados."
