#!/bin/bash
#
# Para os serviços doc-pipeline
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Redireciona para o script principal
exec "$SCRIPT_DIR/start-server.sh" stop
