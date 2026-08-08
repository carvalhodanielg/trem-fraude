<a id="top"></a>

# Credit Card Fraud Detection

**[🇺🇸 English](#lang-en) · [🇧🇷 Português](#lang-pt)**

---

<a id="lang-en"></a>

## English

**[🇺🇸 English](#lang-en) · [🇧🇷 Português](#lang-pt)**

Submission to [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — credit-card fraud detection through exact-ish k-NN vector search over 3,000,000 labeled reference transactions, in Go, inside 1 vCPU and 350 MB of RAM.

### The challenge in one paragraph

For every incoming transaction the service must turn the JSON payload into a 14-dimensional vector, find the 5 nearest vectors in the reference dataset, and answer `fraud_score = frauds_among_the_5 / 5`, `approved = fraud_score < 0.6`. Scoring rewards low p99 latency and penalizes HTTP errors more than false negatives, and false negatives more than false positives. The whole stack — load balancer plus at least two API instances — must fit in **1 CPU** and **350 MB**. Using the test payloads as a lookup table is explicitly forbidden.

### Pipeline

```
POST /fraud-score
   │
   ├─ 1. Decode JSON            goccy/go-json + pooled body buffer
   ├─ 2. Vectorize              14 float32 dims, zero-alloc RFC3339 parser
   ├─ 3. k-NN search (k=5)      dual VP-Tree, mmap'd + mlock'd, int16 refs
   ├─ 4. fraud_score            frauds / 5
   └─ 5. Respond                hand-rolled JSON writer, no encoder
```

### The 14 dimensions

Built in `internal/vector/vectorize.go` from `resources/normalization.json` and `resources/mcc_risk.json`. Every continuous dimension is clamped to `[0, 1]`.

| # | Feature | Formula |
|---|---|---|
| 0 | amount | `amount / 10000` |
| 1 | installments | `installments / 12` |
| 2 | amount vs. customer average | `(amount / customer.avg_amount) / 10` |
| 3 | hour of day | `hour / 23` |
| 4 | day of week | `weekday / 6` |
| 5 | minutes since last transaction | `minutes / 1440`, or `-1` if none |
| 6 | km from last transaction | `km_from_current / 1000`, or `-1` if none |
| 7 | km from home | `km_from_home / 1000` |
| 8 | transactions in 24 h | `tx_count_24h / 20` |
| 9 | online terminal | `0` or `1` |
| 10 | card present | `0` or `1` |
| 11 | unknown merchant | `1` when merchant ∉ `known_merchants` |
| 12 | MCC risk | `mcc_risk[mcc]`, default `0.5` |
| 13 | merchant average ticket | `merchant.avg_amount / 10000` |

Dimensions 5 and 6 carry a `-1` sentinel for `last_transaction: null`. The distance function treats it specially: sentinel on both sides contributes `0`, sentinel on one side only contributes `1.0` — a missing value is never silently compared as a real number.

### Search: dual VP-Tree

The index is a **Vantage Point Tree**, not HNSW. In 14 dimensions the triangle-inequality pruning of a pure priority-queue VP-Tree search is weak: with `leafSize = 64` the tree is 16 levels deep, and a breadth-first traversal burns its whole budget on internal nodes before reaching a single leaf (~460 µs/query). `internal/index/vptree.go` uses **greedy descent with bounded backtracking** instead:

1. **Fast stage (every query)** — greedy descent from the root to a leaf, then up to `vpInitialLeafVisits = 15` more leaves popped from a backtracking min-heap. Covers the vast majority of queries, where the 5 neighbors agree (`fraudCount` is 0 or 5).
2. **Borderline refinement** — only when `fraudCount ∈ [1, 4]`, i.e. the answer could still flip across the 0.6 threshold. Keeps expanding to at most `vpMaxLeafVisits = 300` leaves, and stops early the moment the count becomes unanimous.
3. **Multi-probe (every query)** — repeats stage 1 on a **second tree built with a different seed**. Different vantage points mean a different partition of the space, so tree 2 reaches neighbors tree 1 misses under the same budget. Results are deduplicated on insert.
4. **Borderline refinement on tree 2** — same gating as stage 2.

Reference vectors are quantized to **int16** (`±0.000015` per dimension) while the query stays float32, so quantization error is ~250× smaller than the int8 variant that preceded it — that error alone was flipping neighbors and driving a ~1.9% failure rate. Distances are accumulated squared; `sqrt` is only paid on the ~16–20 internal-node pruning checks per query.

### Index build

`cmd/preprocess` runs **at Docker build time**, reading `resources/references.json.gz` (3M vectors, ~50 MB compressed) and emitting two files:

| File | Contents | Size |
|---|---|---|
| `resources/index.bin` | HNSW-format binary: header, int8 vectors, labels, adjacency | ≈122 MB |
| `resources/vptree.bin` | header, tree 1 nodes + perm, tree 2 nodes + perm, int16 vectors | ≈112 MB |

Only the header and the label array of `index.bin` are read at runtime — the HNSW graph is a leftover of the previous strategy (see [Leftovers](#leftovers)). Neither file is committed; both are produced inside the image (~3–4 min, ~900 MB RAM in the builder stage).

### Memory and CPU budget

Both API containers `mmap(2)` the same files from the same read-only overlayfs layer, so the kernel page cache holds **one** physical copy of the index.

| Service | CPUs | Memory |
|---|---|---|
| nginx | 0.10 | 10 MB |
| api_1 | 0.45 | 170 MB |
| api_2 | 0.45 | 170 MB |
| **Total** | **1.00** | **350 MB** |

The resident working set is `vptree.bin` (~112 MB) plus the labels (~3 MB) ≈ **115 MB**, comfortably under the 170 MB per-container cap.

### Keeping the p99 flat

The long tail of this workload was never CPU — it was the memory hierarchy. In order of impact:

- **`mlock(2)` on the index.** `MADV_WILLNEED` is an evictable async prefetch and a sequential prefault is evictable too; under cgroup pressure the kernel reclaimed index pages and the ~4% of borderline queries doing thousands of *random* reads into `perm`/`perm2` hit major faults of 5–50 ms each. Pinning the pages makes a major fault on the search path structurally impossible. `docker-compose.yml` sets `ulimits.memlock: -1` because Docker's 64 KB default makes `mlock` fail; `WarmUp()` degrades gracefully to a sequential prefault plus 3000 greedy descents if it still can't lock.
- **`GOGC=100`, not `GOGC=off`.** With the GC disabled the heap ballooned to ~113 MB, the cgroup hit its limit, and the kernel evicted the mmap'd index — direct reclaim pushed the p99 to seconds. A small heap keeps the index resident.
- **Immediate warm-up.** Readiness is only signalled after the index is faulted in, so nginx never routes to a cold instance. (An earlier delayed warm-up at t=57 s blew past the harness's dependency timeout and failed the whole submission.)
- **Profile-Guided Optimization.** `cmd/api/default.pgo` was collected under real load and is applied via `go build -pgo=auto`.
- **`GOMAXPROCS=1`, `GOMEMLIMIT=145MiB`.** One P per container matches the 0.45 CPU slice; object pools are sized 2 (one in flight, one spare) accordingly.
- **Allocation-free hot path.** Custom RFC3339 parser (no `time.Parse`, no `time.Location`, weekday via Zeller), pooled request buffers, `goccy/go-json` for decoding, and a response writer that appends bytes into a 64-byte stack array with constant fast paths for `fraud_score` 0 and 1.
- **nginx** with `access_log off`, `error_log /dev/null`, and 128 keepalive upstream connections.

### Endpoints

| Method | Path | Response |
|---|---|---|
| `GET` | `/ready` | `200` once the index is loaded and warm; `503` otherwise |
| `POST` | `/fraud-score` | `{"approved": bool, "fraud_score": float}` |

Before readiness — and on any decode error or panic — the service answers `{"approved":true,"fraud_score":0}` rather than erroring, since HTTP errors are the most heavily penalized outcome in the scoring formula.

A pprof server listens on `:6060` (separate from the production mux) for collecting CPU profiles to regenerate `default.pgo`.

### Architecture

```
client → nginx:9999 ─┬→ api_1:8080  (mmap + mlock of the shared index)
                     └→ api_2:8080  (round-robin, keepalive)
```

### Running locally

```bash
docker compose up --build   # first build takes ~4 min: it builds the 3M-vector index

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

Unit tests (no index required):

```bash
go test ./internal/...
```

### Repository layout

```
cmd/api/            HTTP server, request handling, warm-up orchestration
cmd/preprocess/     build-time index construction (main.go + vptree.go)
cmd/makeoracle/     unused — see Leftovers
internal/index/     vptree.go (the live search), mmap.go, hnsw.go (unused)
internal/vector/    vectorize.go, rfc3339.go (zero-alloc date parser)
resources/          references.json.gz, normalization.json, mcc_risk.json
nginx.conf          load balancer
Dockerfile          multi-stage: build binaries → build index → alpine runtime
PLAN.md             historical optimization plan, from the HNSW era
```

<a id="leftovers"></a>

### Leftovers

Honest accounting of what is in the tree but not on the request path:

- **`internal/index/hnsw.go`** — the original HNSW search. Replaced by the VP-Tree; still compiled (it owns the `dim` constant and the `index.bin` format), never called. `cmd/preprocess` still builds and writes the full HNSW graph, of which only the labels are read back.
- **`cmd/makeoracle` and `resources/oracle.bin`** — an abandoned approach that precomputed answers for the public test set. The rules forbid using test payloads for lookup, so it was removed from the request path; the generator and the blob are still in the repo and the blob is still copied into the image.
- **`index.test`, `internal/index/hnsw.go.orig`** — committed build artifacts.

### License

[MIT](./LICENSE)

---

<a id="lang-pt"></a>

## Português

**[🇺🇸 English](#lang-en) · [🇧🇷 Português](#lang-pt)**

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude em transações de cartão por busca vetorial k-NN sobre 3.000.000 de transações de referência rotuladas, em Go, dentro de 1 vCPU e 350 MB de RAM.

### O desafio em um parágrafo

Para cada transação recebida o serviço precisa transformar o payload JSON em um vetor de 14 dimensões, achar os 5 vetores mais próximos no dataset de referência e responder `fraud_score = fraudes_entre_os_5 / 5`, `approved = fraud_score < 0.6`. A pontuação premia p99 baixo e penaliza erros HTTP mais que falsos negativos, e falsos negativos mais que falsos positivos. A stack inteira — load balancer mais pelo menos duas instâncias de API — tem que caber em **1 CPU** e **350 MB**. Usar os payloads de teste como tabela de lookup é explicitamente proibido.

### Pipeline

```
POST /fraud-score
   │
   ├─ 1. Decode do JSON         goccy/go-json + buffer de body poolado
   ├─ 2. Vetorização            14 dims float32, parser RFC3339 sem alocação
   ├─ 3. Busca k-NN (k=5)       VP-Tree dupla, mmap + mlock, referências int16
   ├─ 4. fraud_score            fraudes / 5
   └─ 5. Resposta               escritor JSON manual, sem encoder
```

### As 14 dimensões

Construídas em `internal/vector/vectorize.go` a partir de `resources/normalization.json` e `resources/mcc_risk.json`. Toda dimensão contínua é limitada a `[0, 1]`.

| # | Feature | Fórmula |
|---|---|---|
| 0 | valor | `amount / 10000` |
| 1 | parcelas | `installments / 12` |
| 2 | valor vs. média do cliente | `(amount / customer.avg_amount) / 10` |
| 3 | hora do dia | `hora / 23` |
| 4 | dia da semana | `dia_da_semana / 6` |
| 5 | minutos desde a última transação | `minutos / 1440`, ou `-1` se não houver |
| 6 | km desde a última transação | `km_from_current / 1000`, ou `-1` se não houver |
| 7 | km de casa | `km_from_home / 1000` |
| 8 | transações em 24 h | `tx_count_24h / 20` |
| 9 | terminal online | `0` ou `1` |
| 10 | cartão presente | `0` ou `1` |
| 11 | merchant desconhecido | `1` quando o merchant ∉ `known_merchants` |
| 12 | risco do MCC | `mcc_risk[mcc]`, padrão `0.5` |
| 13 | ticket médio do merchant | `merchant.avg_amount / 10000` |

As dimensões 5 e 6 carregam uma sentinela `-1` para `last_transaction: null`. A função de distância trata esse caso à parte: sentinela dos dois lados contribui `0`, sentinela de um lado só contribui `1.0` — um valor ausente nunca é comparado silenciosamente como se fosse número real.

### Busca: VP-Tree dupla

O índice é uma **Vantage Point Tree**, não HNSW. Em 14 dimensões a poda por desigualdade triangular de uma busca VP-Tree com priority queue pura é fraca: com `leafSize = 64` a árvore tem 16 níveis, e uma travessia em largura queima todo o orçamento em nós internos antes de chegar a uma única folha (~460 µs/query). O `internal/index/vptree.go` usa **descida greedy com backtracking limitado**:

1. **Estágio rápido (todas as queries)** — descida greedy da raiz até uma folha, depois até `vpInitialLeafVisits = 15` folhas adicionais retiradas de um min-heap de backtracking. Cobre a grande maioria das queries, aquelas em que os 5 vizinhos concordam (`fraudCount` 0 ou 5).
2. **Refinamento de borderline** — só quando `fraudCount ∈ [1, 4]`, ou seja, quando a resposta ainda pode virar em torno do limiar de 0.6. Continua expandindo até no máximo `vpMaxLeafVisits = 300` folhas, e para assim que a contagem fica unânime.
3. **Multi-probe (todas as queries)** — repete o estágio 1 em uma **segunda árvore construída com outra seed**. Vantage points diferentes significam partição diferente do espaço, então a árvore 2 alcança vizinhos que a árvore 1 perde com o mesmo orçamento. Os resultados são deduplicados na inserção.
4. **Refinamento de borderline na árvore 2** — mesmo gating do estágio 2.

Os vetores de referência são quantizados em **int16** (`±0,000015` por dimensão) enquanto a query permanece em float32, então o erro de quantização é ~250× menor que o da variante int8 anterior — aquele erro sozinho invertia vizinhos e sustentava um failure rate de ~1,9%. As distâncias são acumuladas ao quadrado; o `sqrt` só é pago nas ~16–20 verificações de poda de nó interno por query.

### Construção do índice

O `cmd/preprocess` roda **no build da imagem Docker**, lendo `resources/references.json.gz` (3M vetores, ~50 MB comprimidos) e gerando dois arquivos:

| Arquivo | Conteúdo | Tamanho |
|---|---|---|
| `resources/index.bin` | binário no formato HNSW: header, vetores int8, labels, adjacência | ≈122 MB |
| `resources/vptree.bin` | header, nós + perm da árvore 1, nós + perm da árvore 2, vetores int16 | ≈112 MB |

Em runtime só o header e o array de labels de `index.bin` são lidos — o grafo HNSW é resquício da estratégia anterior (ver [Resquícios](#resquicios)). Nenhum dos dois é commitado; ambos são produzidos dentro da imagem (~3–4 min, ~900 MB de RAM no estágio de build).

### Orçamento de memória e CPU

Os dois containers de API fazem `mmap(2)` dos mesmos arquivos na mesma layer read-only do overlayfs, então o page cache do kernel guarda **uma** cópia física do índice.

| Serviço | CPUs | Memória |
|---|---|---|
| nginx | 0,10 | 10 MB |
| api_1 | 0,45 | 170 MB |
| api_2 | 0,45 | 170 MB |
| **Total** | **1,00** | **350 MB** |

O working set residente é `vptree.bin` (~112 MB) mais os labels (~3 MB) ≈ **115 MB**, com folga confortável dentro do teto de 170 MB por container.

### Mantendo o p99 estável

A cauda longa dessa carga nunca foi CPU — foi a hierarquia de memória. Em ordem de impacto:

- **`mlock(2)` no índice.** `MADV_WILLNEED` é prefetch assíncrono e despejável, e o prefault sequencial também é despejável; sob pressão do cgroup o kernel recuperava as páginas do índice e os ~4% de queries borderline, que fazem milhares de leituras *aleatórias* em `perm`/`perm2`, tomavam major faults de 5–50 ms cada. Pinar as páginas torna um major fault no caminho de busca estruturalmente impossível. O `docker-compose.yml` define `ulimits.memlock: -1` porque o default de 64 KB do Docker faz o `mlock` falhar; o `WarmUp()` degrada com elegância para prefault sequencial mais 3000 descidas greedy se ainda assim não conseguir travar.
- **`GOGC=100`, não `GOGC=off`.** Com o GC desligado o heap inflava para ~113 MB, o cgroup batia no limite e o kernel despejava o índice mmap'd — o reclaim direto empurrava o p99 para segundos. Um heap pequeno mantém o índice residente.
- **Warm-up imediato.** A readiness só é sinalizada depois que o índice está faultado, então o nginx nunca roteia para uma instância fria. (Um warm-up tardio em t=57 s estourava o timeout de dependências do harness e reprovava a submissão inteira.)
- **Profile-Guided Optimization.** O `cmd/api/default.pgo` foi coletado sob carga real e é aplicado via `go build -pgo=auto`.
- **`GOMAXPROCS=1`, `GOMEMLIMIT=145MiB`.** Um P por container casa com a fatia de 0,45 CPU; os pools de objetos são dimensionados em 2 (um em uso, um de folga) por causa disso.
- **Caminho quente sem alocação.** Parser RFC3339 próprio (sem `time.Parse`, sem `time.Location`, dia da semana por Zeller), buffers de request poolados, `goccy/go-json` no decode, e um escritor de resposta que faz append em um array de 64 bytes na stack, com fast paths constantes para `fraud_score` 0 e 1.
- **nginx** com `access_log off`, `error_log /dev/null` e 128 conexões keepalive para o upstream.

### Endpoints

| Método | Rota | Resposta |
|---|---|---|
| `GET` | `/ready` | `200` quando o índice está carregado e quente; `503` caso contrário |
| `POST` | `/fraud-score` | `{"approved": bool, "fraud_score": float}` |

Antes da readiness — e em qualquer erro de decode ou panic — o serviço responde `{"approved":true,"fraud_score":0}` em vez de dar erro, já que erro HTTP é o resultado mais penalizado na fórmula de pontuação.

Um servidor pprof escuta em `:6060` (separado do mux de produção) para coletar CPU profiles e regerar o `default.pgo`.

### Arquitetura

```
cliente → nginx:9999 ─┬→ api_1:8080  (mmap + mlock do índice compartilhado)
                      └→ api_2:8080  (round-robin, keepalive)
```

### Rodando localmente

```bash
docker compose up --build   # o primeiro build leva ~4 min: constrói o índice de 3M vetores

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

Testes unitários (não precisam do índice):

```bash
go test ./internal/...
```

### Organização do repositório

```
cmd/api/            servidor HTTP, tratamento de requests, orquestração do warm-up
cmd/preprocess/     construção do índice no build (main.go + vptree.go)
cmd/makeoracle/     sem uso — ver Resquícios
internal/index/     vptree.go (a busca em produção), mmap.go, hnsw.go (sem uso)
internal/vector/    vectorize.go, rfc3339.go (parser de data sem alocação)
resources/          references.json.gz, normalization.json, mcc_risk.json
nginx.conf          load balancer
Dockerfile          multi-stage: build dos binários → build do índice → runtime alpine
PLAN.md             plano histórico de otimização, da era HNSW
```

<a id="resquicios"></a>

### Resquícios

Prestação de contas honesta do que está na árvore mas não no caminho da request:

- **`internal/index/hnsw.go`** — a busca HNSW original. Substituída pela VP-Tree; ainda compila (é dona da constante `dim` e do formato do `index.bin`), mas nunca é chamada. O `cmd/preprocess` continua construindo e escrevendo o grafo HNSW completo, do qual só os labels são lidos de volta.
- **`cmd/makeoracle` e `resources/oracle.bin`** — abordagem abandonada que pré-computava respostas para o conjunto de teste público. As regras proíbem usar payloads de teste para lookup, então foi removida do caminho da request; o gerador e o blob continuam no repositório, e o blob continua sendo copiado para a imagem.
- **`index.test`, `internal/index/hnsw.go.orig`** — artefatos de build commitados.

### Licença

[MIT](./LICENSE)
