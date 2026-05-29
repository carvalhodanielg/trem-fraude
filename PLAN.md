# PLAN.md — Trem-de-Fraude: regressão + melhorias

## Context

Submissão para Rinha de Backend 2026 (k-NN em 3M vetores 14-D, prazo 2026-06-05). Stack: Go + nginx, HNSW custom (`int8` vectors, `adj0` uint24 LE, M=3, two-stage ef=30/60, pools sync de bitset/heap).

**Estado atual (regressão vs. melhor histórico):**

| | atual | melhor (commit 5970289) |
|---|---|---|
| `final_score` | **1160,59** | 1632–1646 |
| `p99` | 135,65 ms | 80–83 ms |
| FN | **812** | 307–310 |
| FP | 164 | 711–712 |
| `failure_rate` | 1,81% | ~1,89% |
| massa de teste (N) | 54.100 | menor |

O shift FN↑/FP↓ mostra classificação **mais permissiva** (aprova mais fraudes). FN pesa 3× mais que FP no scoring — é o eixo que mais dói.

**Achado crítico antes de qualquer outra coisa:**
`docker-compose.yml` declara `220 + 220 + 10 = 450 MB` de memória total. Regra da Rinha (`.docs/ARQUITETURA.md`): **350 MB máximo**. Em ambiente oficial pode invalidar submissão ou disparar OOM-killer.

**Abordagem aprovada:** caminho balanceado (memória → RFC3339 → JSON → rerank float32 → heap inline → stage-2 continue → SIMD/PGO), cada passo medido isoladamente. Dependências aprovadas: `easyjson` e `golang.org/x/sys/unix`. `mailru/easyjson` é a escolha padrão por gerar código a partir de tags já presentes nos structs.

**Como medir cada passo:**
- Bench Go unitário do hot path (`go test -bench=. -benchmem ./internal/...`).
- Profile sob k6 local (cpu + heap pprof).
- Teste de carga oficial: `docker compose --profile test up` conforme `.docs/AVALIACAO.md`.
- Resultado registrado em `~/.claude/projects/-home-daniel-projetos-trem-de-fraude/memory/hnsw-tuning-results.md` para preservar histórico.

---

## Task 1 — [CRÍTICO] Compartilhar índice via mmap e ajustar limites do compose

### Por que primeiro
A submissão pode ser invalidada por exceder 350 MB. Qualquer otimização posterior é inútil sem isso. Também desbloqueia headroom para futuras melhorias (rerank float32 cabe).

### Mudanças
1. **Novo arquivo `internal/index/mmap.go`:**
   - Função `MmapBytes(path string) ([]byte, error)` usando `golang.org/x/sys/unix.Mmap` com `PROT_READ | MAP_SHARED`.
   - `MAP_POPULATE` para evitar page faults na primeira requisição (opcional; testar trade-off).
   - Retornar slice direto sobre as páginas mapeadas, sem cópia.

2. **`internal/index/hnsw.go::Load`:**
   - Substituir `bufio.NewReaderSize` por leitura via mmap.
   - Construir slices `vectors []int8`, `adj0 []byte`, `labels []uint8`, `adj0Cnt []uint8` como **views** do mmap (usando `unsafe.Slice` com cuidado para alinhar tipos).
   - Camadas superiores (CSR): podem ficar como views do mmap também — são leitura pura.
   - Reduzir `poolSize` de visited de 32 → 4 e `heapPoolSize` de 32 → 4 (com GOMAXPROCS=1, 2 já basta; 4 dá folga).
   - **NÃO copiar** para `make([]int8, ...)` — esse é o ponto de não duplicar.

3. **`Dockerfile`:**
   - Mantém build do índice na imagem (rápido, determinístico).
   - Imagem final continua copiando `resources/index.bin`.

4. **`docker-compose.yml`:**
   - Montar volume read-only compartilhado entre `api_1` e `api_2`:
     - Estratégia A (recomendada): adicionar serviço `init` que copia `/app/resources/index.bin` da imagem para um volume nomeado, ambos api_1/api_2 montam o volume como `:ro`.
     - Estratégia B: usar `tmpfs` com `cp` no entrypoint — menos elegante.
   - Ajustar limites:
     - nginx: `cpus: 0.05, memory: 10MB`
     - api_1: `cpus: 0.475, memory: 160MB`
     - api_2: `cpus: 0.475, memory: 160MB`
     - **Total: 330 MB** (folga de 20 MB)

### Validação
- `docker stats` durante o teste de carga: cada api ≤ 160 MB.
- `cat /sys/fs/cgroup/memory.stat` dentro do container: `mapped_file` significativo, `rss` baixo.
- Teste oficial roda sem OOM e mantém score equivalente ao baseline.

### Risco
Médio. mmap com slice via `unsafe.Slice` exige cuidado com alinhamento (`adj0Cnt []uint8` ok; `[]int8` alinhamento 1; `[]uint32` alinhamento 4 — verificar offset no arquivo). Se complicar, **fallback:** manter `Load` atual mas reduzir o índice (vide Task 8).

---

## Task 2 — [LATÊNCIA] Parser RFC3339 dedicado

### Por que segundo
`time.Parse` aparece 2× em `Vectorize` (linhas 98, 107 de `internal/vector/vectorize.go`). É notoriamente alocante (cria `time.Location`, faz reflection no layout). Ganho típico em hot path: 1–3 µs/req. Risco mínimo, totalmente testável em unit test.

### Mudanças
1. **Novo arquivo `internal/vector/rfc3339.go`:**
   - Formato fixo conhecido: `YYYY-MM-DDTHH:MM:SSZ` ou `YYYY-MM-DDTHH:MM:SS.sssZ`.
   - Funções:
     - `parseHourWeekdayUTC(s string) (hour int, dow int, ok bool)` — extrai direto sem `time.Time`.
     - `parseMinutesBetween(a, b string) (minutes float32, ok bool)` — converte ambos para epoch e subtrai.
   - Parse manual de inteiros via loop (não `strconv.Atoi`, que ainda aloca).
   - Dia da semana via algoritmo de Zeller (sem `time.Weekday()`).

2. **`internal/vector/vectorize.go::Vectorize`:**
   - Substituir `time.Parse(time.RFC3339, ...)` pelas funções novas.
   - Manter fallback em caso de string inválida (devolver 0 para `hour`, `dow`, `minutes`).

3. **Bench:**
   - `BenchmarkParseRFC3339` comparando com `time.Parse` — esperar 10–20× mais rápido.
   - `BenchmarkVectorize` antes/depois — esperar 30–50% mais rápido.

### Validação
- `vectorize_test.go` precisa cobrir casos com `last_transaction: null`, `Z`, com milissegundos, primeira/última hora do dia, segunda/domingo.
- Teste oficial: p99 deve cair alguns ms.

### Risco
Baixo. Parser ad hoc, mas formato é fixo e curto.

---

## Task 3 — [LATÊNCIA] JSON sem reflexão (easyjson)

### Por que terceiro
`json.NewDecoder(r.Body).Decode(&req)` em `cmd/api/main.go:125` usa reflection em cada request. easyjson gera `MarshalJSON/UnmarshalJSON` específicos. Ganho típico: 5–10× em decode, ~µs por request.

### Mudanças
1. **Adicionar dep:** `github.com/mailru/easyjson` no `go.mod`.

2. **`cmd/api/main.go`:**
   - Adicionar comentário `//easyjson:json` acima de `fraudRequest`.
   - Pode também marcar os types do package `vector` (`Transaction`, `Customer`, etc.), mas como o response é manual, só `fraudRequest` é hot path.

3. **Gerar:** `go generate ./...` com diretiva `//go:generate easyjson -all main.go` (commitar o `_easyjson.go` gerado).

4. **Substituir decode:**
   - De: `json.NewDecoder(r.Body).Decode(&req)`
   - Para: ler body em buffer (`sync.Pool` de `[]byte`), depois `req.UnmarshalJSON(buf)`.

5. **Bench:** novo `BenchmarkDecodeFraudRequest` com payload realista de `.docs/example-payloads.json` ou `resources/example-references.json`.

### Validação
- Unit test confirmando que campos batem com `encoding/json`.
- Bench mostra redução de allocs/op para próximo de 0 (struct é stack-allocable).
- Teste oficial mostra p99 caindo.

### Risco
Baixo–médio. easyjson é estável e amplamente usado. Cuidado: ponteiros opcionais (`*LastTransaction`) precisam estar corretamente tagged.

---

## Task 4 — [DETECÇÃO] Rerank float32 do top-K final

### Por que agora
O resultado atual tem FN=812 (regressão). A hipótese mais provável é **perda de resolução por quantização int8** nos vetores: 14 dims densas em `[0,127]` colam muitos vizinhos. Rerankar os top-K (k=5 do stage-2) com distância float32 exata custa quase nada e pode resgatar 200+ FN.

### Mudanças
1. **`cmd/preprocess/main.go`:**
   - Escrever também `vectorsF32 []float32` (N × 14 × 4 bytes = **168 MB**) no fim do `index.bin` — ou em arquivo separado `index_f32.bin`.
   - **Atenção:** 168 MB **excede** o orçamento de memória. **Avaliar antes:**
     - Opção A: usar `bfloat16` (2 bytes/dim → 84 MB) via tipo custom.
     - Opção B: `float16` (`mlas` ou implementação custom) → 84 MB.
     - Opção C: armazenar **apenas dims contínuas** em float32 (skip índices 9, 10, 11 — binárias). 11 dims × 4 × 3M = **132 MB**.
     - Opção D: nada extra; rerankar usando os mesmos vetores int8 com fórmula mais precisa (resolução fracionária na query).
   - **Recomendação:** começar por Opção D (zero memória extra) — se ganho insuficiente, ir para Opção C.

2. **`internal/index/hnsw.go::Search`:**
   - Após stage-2, em vez de devolver `fraudCount/k` direto, calcular distância float32 entre query float (já temos `vec [14]float32` em `Vectorize`) e cada um dos k=5 resultados.
   - Reordenar pelos k=5 menores em float32.
   - Para Opção D: refazer `distSq` com query em float32 e refs em int8 promovido (sem perda no lado da query).

3. **Bench:**
   - Rodar teste oficial com a config atual + rerank. Esperar FN caindo significativamente (alvo: < 500).

### Validação
- Esperado: FN cai, FP pode subir ligeiramente (rerank pode reordenar limítrofes). FN×3 + FP×1 deve cair no agregado.
- `failure_rate` deve permanecer < 2%.
- p99 deve subir < 0,5 ms.

### Risco
Baixo se Opção D. Médio se C/B (precisa garantir limite de memória — só viável após Task 1).

---

## Task 5 — [LATÊNCIA] Heap inline em `beamSearchL0`

### Por que
`container/heap` é via `interface{}` → boxing por elemento (`heap.Push(cands, cand{...})` aloca). Beam search faz milhares de push/pop. Ganho típico: 2–3× no beam search.

### Mudanças
1. **`internal/index/hnsw.go`:**
   - Remover `minCandHeap`/`maxCandHeap` baseados em `heap.Interface`.
   - Implementar funções inline `pushMin(h *[]cand, c cand)`, `popMin(h *[]cand) cand`, `pushMax`, `popMax` — bubble up/down direto sobre o slice.
   - Sem `any`, sem boxing, sem dispatch dinâmico.

2. **Bench:**
   - `BenchmarkBeamSearchL0` antes/depois.
   - Profile com pprof: confirmar que `runtime.convT` desaparece do top.

### Validação
- Unit tests existentes em `hnsw_test.go` precisam passar (corretude do beam search).
- Bench mostra redução substancial de allocs/op.

### Risco
Baixo. Mudança mecânica, totalmente coberta por testes.

---

## Task 6 — [LATÊNCIA] Stage-2 continua o stage-1

### Por que
`Search` (`hnsw.go:376`) refaz beam search do zero quando entra no stage-2 (10–20% das requests, casos limítrofes). Reusar `visited` e `res` do stage-1 e apenas elevar `ef` evita trabalho duplicado.

### Mudanças
1. **`internal/index/hnsw.go::beamSearchL0`:**
   - Adicionar parâmetro `continueFrom *heapBufs` ou flag `expand bool`.
   - Quando expandindo, **não** zera `hb.cands`, `hb.res`, `visited` — apenas continua o loop com `ef` maior.

2. **`Search`:**
   - Se `fraudCount in {2,3}`, chamar `beamSearchL0(..., ef=efSearch, continueFrom=hb)` em vez de novo beam search.

3. **Bench:**
   - `BenchmarkSearchAmbiguous` (gerar queries no limite) — esperar 30–50% mais rápido nos casos de stage-2.

### Validação
- Unit test que confirma resultados iguais aos do stage-2 atual.
- Teste oficial: p99 cai nos percentis altos (95+).

### Risco
Médio. Lógica do beam search com estado preservado é sutil (precisa garantir que candidatos descartados pelo ef=30 voltem ao `cands` quando o budget aumenta para 60 — ou aceitar perda mínima de recall).

---

## Task 7 — [LATÊNCIA] Tuning de runtime + PGO

### Por que
GC-tuning sob orçamento de memória apertado vale alguns ms no p99. PGO no Go 1.20+ otimiza branches do hot path com profile real.

### Mudanças
1. **`docker-compose.yml`:**
   - Adicionar env: `GOMEMLIMIT=140MiB`, `GOGC=200`.
2. **PGO:**
   - Adicionar endpoint `/debug/pprof/profile?seconds=30` em build de dev.
   - Coletar profile sob k6, salvar como `default.pgo`.
   - `Dockerfile`: passar `-pgo=default.pgo` ao `go build`.

### Validação
- Teste oficial: ganho marginal 5–15% no p99.
- `runtime/metrics` durante teste confirma pause GC < 1 ms.

### Risco
Baixo.

---

## Task 8 — [FALLBACK] Caso Task 1 (mmap) não dê certo

### Quando aplicar
Apenas se mmap mostrar problemas de alinhamento, contabilização incorreta de `shared` no cgroup do ambiente da Rinha, ou se o ganho real for menor que o esperado.

### Mudanças alternativas
- Quantização int4 dos vetores (14 dims × 4 bits = 7 bytes/nó → 21 MB). Custa shift/mask no distSq.
- Reduzir pool de bitset de 32 → 2 (já confirmado pela memória que não muda performance).
- Reduzir `M` para 2 (perde recall — não recomendado).
- Single instance + faking via nginx (proibido pelas regras).

---

## Arquivos críticos por task

| Task | Arquivos |
|---|---|
| 1 | `internal/index/mmap.go` (novo), `internal/index/hnsw.go::Load`, `docker-compose.yml`, `Dockerfile` |
| 2 | `internal/vector/rfc3339.go` (novo), `internal/vector/vectorize.go`, `internal/vector/vectorize_test.go` |
| 3 | `cmd/api/main.go`, `cmd/api/main_easyjson.go` (gerado), `go.mod` |
| 4 | `internal/index/hnsw.go::Search`, possivelmente `cmd/preprocess/main.go` |
| 5 | `internal/index/hnsw.go` (heap inline) |
| 6 | `internal/index/hnsw.go::beamSearchL0`, `internal/index/hnsw.go::Search` |
| 7 | `Dockerfile`, `docker-compose.yml`, `default.pgo` (novo) |

---

## Sucesso final esperado (alvo)

| | atual | alvo realista | alvo otimista |
|---|---|---|---|
| `final_score` | 1160 | **1800+** | 2500+ |
| `p99` | 135 ms | 50 ms | 20 ms |
| FN | 812 | 400 | 250 |
| memória total | 450 MB ❌ | 330 MB ✅ | 280 MB ✅ |

Cada task entrega de forma incremental, com bench e teste oficial registrados no `hnsw-tuning-results.md`.
