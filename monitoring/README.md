# Monitoring - Doc Pipeline

Configuracoes de monitoramento para o servico doc-pipeline.

## Estrutura

```
monitoring/
├── grafana/
│   ├── dashboards/
│   │   └── doc-pipeline.json    # Dashboard principal
│   └── alerts/
│       └── doc-pipeline-alerts.yaml  # Regras de alerta
├── prometheus/
│   └── prometheus.yml           # Configuracao de scraping
└── README.md
```

## Metricas Disponiveis

| Metrica | Tipo | Labels | Descricao |
|---------|------|--------|-----------|
| `doc_pipeline_requests_total` | Counter | method, endpoint, status | Total de requests |
| `doc_pipeline_request_duration_seconds` | Histogram | method, endpoint | Latencia dos requests |
| `doc_pipeline_requests_in_progress` | Gauge | method, endpoint | Requests em andamento |
| `doc_pipeline_errors_total` | Counter | method, endpoint, error_type | Total de erros |
| `doc_pipeline_documents_processed_total` | Counter | document_type, operation | Docs processados |
| `doc_pipeline_classification_confidence` | Histogram | document_type | Confianca das classificacoes |
| `doc_pipeline_requests_by_client_total` | Counter | client, endpoint, status | Requests por API key/cliente |
| `doc_pipeline_gpu_memory_used_bytes` | Gauge | - | Memoria GPU (se habilitado) |

## Configuracao do Prometheus

1. Copie ou monte `prometheus/prometheus.yml`
2. Ajuste o target para o endereco da API:
   ```yaml
   - targets: ['seu-host:8000']
   ```

## Importar Dashboard no Grafana

### Via UI
1. Grafana > Dashboards > Import
2. Upload `grafana/dashboards/doc-pipeline.json`
3. Selecione o datasource Prometheus

### Via Provisioning
```bash
# Copie para o diretorio de provisioning
cp grafana/dashboards/doc-pipeline.json /etc/grafana/provisioning/dashboards/
```

## Configurar Alertas

### Via Script (recomendado)
```bash
# Cria API token no Grafana:
# Administration > Service accounts > Add service account > Add token

# Opcao 1: Configura no .env (recomendado)
echo "GRAFANA_URL=https://seu-grafana.com" >> .env
echo "GRAFANA_TOKEN=glsa_xxxxxxxxxxxx" >> .env
./monitoring/scripts/create-alerts.sh

# Opcao 2: Passa como argumentos
./monitoring/scripts/create-alerts.sh https://seu-grafana.com glsa_xxxxxxxxxxxx

# Opcao 3: Com datasource UID manual
./monitoring/scripts/create-alerts.sh https://seu-grafana.com glsa_xxx abc123def456
```

### Via Provisioning (requer acesso ao servidor)
```bash
cp grafana/alerts/doc-pipeline-alerts.yaml /etc/grafana/provisioning/alerting/
```

### Alertas Configurados

| Alerta | Severidade | Condicao |
|--------|------------|----------|
| High Error Rate (5xx) | critical | >5% erros 5xx em 5min |
| High Client Error Rate (4xx) | warning | >20% erros 4xx em 10min |
| No Traffic | critical | 0 requests em 10min |
| High Latency P95 | warning | P95 > 30s por 5min |
| Critical Latency P99 | critical | P99 > 60s por 5min |
| High Request Concurrency | warning | >10 requests simultaneos |
| Low Classification Confidence | warning | Mediana < 0.7 por 15min |
| Very Low Classification Confidence | critical | Mediana < 0.5 por 5min |
| Model/Inference Errors | critical | Erros CUDA/RuntimeError |
| SLO: Availability Below Target | critical | Disponibilidade < 99% (1h) |
| SLO: Latency Below Target | warning | <95% requests < 30s (1h) |

## Queries Uteis

### Taxa de erro por endpoint
```promql
sum by (endpoint) (rate(doc_pipeline_requests_total{status=~"5.."}[5m]))
/
sum by (endpoint) (rate(doc_pipeline_requests_total[5m]))
```

### Latencia P95 por endpoint
```promql
histogram_quantile(0.95,
  sum by (endpoint, le) (rate(doc_pipeline_request_duration_seconds_bucket[5m]))
)
```

### Documentos processados por tipo (ultimas 24h)
```promql
sum by (document_type) (increase(doc_pipeline_documents_processed_total[24h]))
```

### Confianca media por tipo de documento
```promql
histogram_quantile(0.5,
  sum by (document_type, le) (rate(doc_pipeline_classification_confidence_bucket[5m]))
)
```

### SLO de disponibilidade (99% target)
```promql
1 - (
  sum(rate(doc_pipeline_requests_total{status=~"5.."}[1h]))
  /
  sum(rate(doc_pipeline_requests_total[1h]))
)
```

### Throughput por operacao
```promql
sum by (operation) (rate(doc_pipeline_documents_processed_total[5m]))
```

### Requests por cliente (ultimas 24h)
```promql
sum by (client) (increase(doc_pipeline_requests_by_client_total[24h]))
```

### Top 5 clientes por uso
```promql
topk(5, sum by (client) (rate(doc_pipeline_requests_by_client_total[1h])))
```

### Taxa de erro por cliente
```promql
sum by (client) (rate(doc_pipeline_requests_by_client_total{status=~"4..|5.."}[5m]))
/
sum by (client) (rate(doc_pipeline_requests_by_client_total[5m]))
```

## Integracao com Alertmanager

Configure receivers no Alertmanager para notificacoes:

```yaml
# alertmanager.yml
route:
  receiver: 'default'
  routes:
    - match:
        service: doc-pipeline
        severity: critical
      receiver: 'pagerduty'
    - match:
        service: doc-pipeline
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/xxx'
        channel: '#alerts'
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'xxx'
```

## Docker Compose (exemplo)

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/alerts:/etc/grafana/provisioning/alerting
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```
