# trem-de-fraude

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude em transações de cartão usando busca vetorial.

## Como funciona

Para cada transação recebida, a API:

1. **Vetoriza** o payload em 14 dimensões (amount, installments, hour of day, km from home, etc.)
2. **Busca** no dataset de referência (3M transações rotuladas) os 5 vizinhos mais próximos via índice HNSW
3. **Calcula** `fraud_score = fraudes_entre_os_5 / 5`
4. **Responde** `approved = fraud_score < 0.6`

## Stack

- **Go** — API HTTP + vetorização + busca HNSW
- **nginx** — load balancer (round-robin entre 2 instâncias)

## Endpoints

```
GET  /ready        → 200 quando o índice estiver carregado
POST /fraud-score  → { "approved": bool, "fraud_score": float }
```

## Arquitetura

```
cliente → nginx:9999 → api_1:8080
                     → api_2:8080
```

Limites: 1 vCPU e 350 MB de RAM no total (conforme regras da Rinha).

## Índice vetorial

O índice HNSW é construído no build da imagem Docker (`cmd/preprocess`) a partir de `resources/references.json.gz` (3M vetores). Usa quantização int8 e formato binário compacto (~134 MB), cabendo dentro do limite de memória com folga para o runtime.

## Rodando localmente

```bash
# Build e start
docker compose up --build

# Testar
curl http://localhost:9999/ready

curl -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tx-123",
    "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T18:45:53Z" },
    "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-001"] },
    "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
    "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.7 },
    "last_transaction": null
  }'
```

## Licença

[MIT](./LICENSE)
