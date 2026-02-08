# Post-Mortem: O Worker Fantasma (2026-02-08)

## TL;DR
Um container `docker compose run` esquecido ha 47 horas roubava jobs da fila Redis com codigo antigo e falhava silenciosamente com "Too many connections". Levou ~6 horas de debug para encontrar porque `docker compose logs` **nao mostra containers criados por `docker compose run`**.

## Sintoma
Load test com 400 requests / 170 concurrency: ~30% dos requests retornavam HTTP 500 com `"Too many connections"`.

## O que tornava dificil
1. **Workers nao logavam o erro** — os 5 workers gerenciados pelo `docker compose up` nunca mostravam "Too many connections" nos logs
2. **O erro vinha de dentro do resultado cached** — o API recebia o resultado via Redis cache (`result:{request_id}`) ja com `error: "Too many connections"`, como se o worker tivesse processado e falhado
3. **Timing impossivel** — jobs apareciam "processados" em 218ms, quando o processamento normal leva ~7s

## Hipoteses testadas (todas erradas)
| # | Hipotese | Acao | Resultado |
|---|----------|------|-----------|
| 1 | Pub/Sub do API esgotando conexoes | Migrou `wait_for_result()` de Pub/Sub para polling | Erro persistiu |
| 2 | slowapi criando pool Redis proprio | Mudou limiter para `memory://` quando desabilitado | Erro persistiu |
| 3 | Workers com codigo antigo (sem BlockingConnectionPool) | Rebuild de todos os 5 workers | Erro persistiu |
| 4 | InferenceClient usando Pub/Sub | Converteu inference_client.py para polling | Erro persistiu |
| 5 | `max_connections=20` muito baixo nos workers | Aumentou default para 200 | Erro persistiu |
| 6 | Bug no BlockingConnectionPool do redis-py 7.1.0 | Analisou source code | Possivel mas nao root cause |
| 7 | Erro vindo da Nebius (cloud provider) | Descartado via analise | N/A |

## A sacada
Adicionamos logging diagnostico que capturava o payload completo do resultado cached:
```json
{
  "started_at": "15:40:03.216",
  "completed_at": "15:40:03.435",
  "processing_time_ms": 218,
  "error": "Too many connections"
}
```
**218ms de processamento** para algo que leva 7s. E **nenhum** worker logava este request_id.

Isso levou a verificar `docker ps` com mais atencao:
```
doc-pipeline-worker-docid-1-run-20b2282c3e52   Up 47 hours   77204189c827
```

Um container `docker compose run` esquecido, rodando com imagem de 2 dias atras.

## Confirmacao
```bash
docker logs doc-pipeline-worker-docid-1-run-20b2282c3e52 --since 5m | grep "too many"
# DEZENAS de "Too many connections" — era ele o tempo todo!
```

## Root cause
- Alguem (provavelmente durante debug) rodou `docker compose run worker-docid-1` manualmente
- Esse container ficou vivo consumindo da mesma fila `queue:doc:documents`
- Usava codigo antigo: `ConnectionPool` regular com `max_connections=20`
- Sob carga, esgotava suas 20 conexoes rapidamente
- Falhava com "Too many connections", cacheava o erro no Redis, e seguia roubando mais jobs
- `docker compose logs` nunca mostra containers de `run` — so `up`

## Fix
```bash
docker stop doc-pipeline-worker-docid-1-run-20b2282c3e52
docker rm doc-pipeline-worker-docid-1-run-20b2282c3e52
```

**Resultado: 400/400 OK, 0 falhas.**

## Melhorias implementadas durante o debug (que ficaram)
1. **Pub/Sub -> Polling** no `wait_for_result()` — elimina conexoes persistentes, muito mais escalavel
2. **Pub/Sub -> Polling** no `InferenceClient` — mesmo beneficio
3. **BlockingConnectionPool** com `max_connections=200` em todos os servicos
4. **slowapi usa `memory://`** quando rate limit desabilitado — zero conexoes Redis desperdicadas
5. **Workers 1-5 sempre ativos** (sem autoscaler) — workers sao leves (~800MB) com inference server centralizado

## Licoes aprendidas

### 1. `docker compose logs` tem ponto cego
`docker compose logs` so mostra containers do `docker compose up`. Containers de `docker compose run` sao invisiveis. Use `docker ps` + `docker logs <nome>` para ver tudo.

### 2. Sempre cheque `docker ps` por containers orfaos
Antes de debugar, rodar `docker ps | grep <project>` e verificar se ha containers inesperados.

### 3. O timing do erro cached conta uma historia
218ms de "processing_time" quando o normal e 7s = o job nao foi processado de verdade. Isso deveria ter sido o primeiro sinal.

### 4. Tags de debug sao eficazes mas limitados
Taguear cada `raise HTTPException` com identificadores unicos (#P1, #P2, etc.) foi eficaz para provar que o erro NAO vinha de nenhum handler do API. Mas nao ajuda quando o erro vem de FORA (resultado cached por outro processo).

### 5. Trace pelo request_id e rei
Grepar um request_id especifico em TODOS os logs de TODOS os servicos e a forma mais rapida de identificar quem processou (ou nao) um job. Se nenhum worker logou aquele ID, alguem mais esta consumindo da fila.
