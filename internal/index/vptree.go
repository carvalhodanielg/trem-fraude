// Package index — VP-Tree (Vantage Point Tree) para busca k-NN.
//
// Estratégia: greedy descent + backtracking limitado.
//
// O problema com busca por priority-queue pura em 14D:
//   A poda por triangle inequality é fraca (maldição da dimensionalidade).
//   Para N=3M/leafSize=64 a árvore tem profundidade 16. Uma busca BFS
//   precisa processar 2^16−1=65535 nós internos para chegar nas folhas.
//   Com um orçamento razoável (5000 operações), nunca chegamos às folhas.
//
// Solução: greedy descent + backtracking limitado.
//   Fase 1: descer greedy da raiz até a primeira folha (16 internos + 64 pontos).
//   Fase 2: explorar vpMaxLeafVisits−1 folhas adicionais via PQ de backtracking.
//   Total: ~350 operações × ~7ns = ~2.5µs por query (vs ~460µs com PQ puro).
//
// Precisão: float32 query vs int16 referência — 250× mais preciso que int8.
//   Erro L2 máximo por quantização int16: sqrt(14 × (1/65534)²) ≈ 0.000028.
//   Elimina virtualmente todos os erros de classificação por quantização.
//   O int8 original (±0.004/dim) causava ~1.9% de failure_rate por flip de vizinhos.
//
// Formato dos arquivos:
//
//	index.bin: mesmo formato HNSW (reutilizado apenas para labels)
//	vptree.bin: header(16B) + nodes(nodeCount×16B) + perm(N×4B) + vectors16(N×14×2B)
//
// Codificação de folha em vpNode.left/right:
//
//	left ≥ 0  → nó interno, filho esquerdo em nodes[left]
//	left < 0  → folha: lo = -(left+1), hi = -(right+1), range perm[lo:hi]
package index

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"time"
	"unsafe"
)

// vpInitialLeafVisits: folhas visitadas na busca rápida (todos os casos).
// ~5×(16+64) ≈ 400 ops × 7ns ≈ 2.8µs/query.
const vpInitialLeafVisits = 5

// vpMaxLeafVisits: folhas visitadas TOTAL na busca refinada (só casos borderline).
// ~700×(16+64) ≈ 56000 ops × 7ns ≈ 392µs/query (apenas ~4% dos casos).
// Média ponderada: 0.96×2.8µs + 0.04×392µs ≈ 18.4µs/query.
const vpMaxLeafVisits = 700

// vpPQInitialCap: capacidade inicial do PQ de backtracking.
// Com max_visits=1000 e depth=16: até 1000×16=16000 entradas.
// Pré-alocar evita realocações durante a busca (custo amortizado zero).
const vpPQInitialCap = 16384

// vpNode é um nó da VP-Tree, armazenado em vptree.bin (16 bytes cada).
//
// Nó interno: vpIdx válido, medDist em distância real (não ao quadrado),
//
//	left ≥ 0, right ≥ 0.
//
// Nó folha:   left = -(lo+1) < 0, right = -(hi+1) < 0.
//
//	Recuperação: lo = -(left+1), hi = -(right+1)
type vpNode struct {
	vpIdx   uint32  // índice do vantage point nos arrays vectors/labels
	medDist float32 // distância mediana real (L2, não ao quadrado)
	left    int32   // filho esquerdo (≥0) ou -(lo+1) se folha
	right   int32   // filho direito  (≥0) ou -(hi+1) se folha
}

// vpKNN é um max-heap de até k=5 vizinhos mais próximos.
// Armazena distâncias AO QUADRADO — evita sqrt em pontos de folha.
// sqrt só é chamado no pruning de nós internos (~16-20 vezes/query).
type vpKNN struct {
	dSq [5]float32 // distâncias ao quadrado (sem sqrt)
	idx [5]uint32  // índices originais dos pontos
	n   int        // quantidade preenchida (0 ≤ n ≤ 5)
}

// vpPQEntry é uma entrada na priority queue de backtracking.
// minDist é a distância REAL (necessária para triangle inequality).
type vpPQEntry struct {
	nodeIdx int32   // índice em VPIndex.nodes
	minDist float32 // menor distância real possível para qualquer ponto nessa subárvore
}

// vpPQ é uma min-heap de vpPQEntry, ordenada por minDist.
type vpPQ struct {
	buf []vpPQEntry
}

// VPIndex é o índice VP-Tree carregado via mmap.
// Duas árvores (seeds 1 e 2) estão presentes no vptree.bin; a busca usa tree 1.
// Tree 2 é carregada mas não usada na busca (reservada para experimentos futuros).
type VPIndex struct {
	n         int
	vectors16 []int16 // mmap'd de vptree.bin: N×dim int16 (±0.000015/dim)
	labels    []uint8 // mmap'd de index.bin: N uint8 (0=legit, 1=fraud)
	// Árvore 1 (seed=1)
	nodes []vpNode // mmap'd de vptree.bin
	perm  []uint32 // mmap'd de vptree.bin
	// Árvore 2 (seed=2) — mesmos vetores, estrutura diferente (~13.5MB extra)
	nodes2 []vpNode // mmap'd de vptree.bin
	perm2  []uint32 // mmap'd de vptree.bin

	// Pools de tamanho 2: com GOMAXPROCS=1, 1 em uso + 1 de folga.
	knnPool  chan *vpKNN
	pqPool   chan *vpPQ // PQ para árvore 1
	pq2Pool  chan *vpPQ // PQ para árvore 2

	mmapIdx  []byte // mantém index.bin vivo (mmap)
	mmapTree []byte // mantém vptree.bin vivo (mmap)
}

// LoadVP carrega o índice dual VP-Tree a partir de dois arquivos:
//   - indexPath: index.bin existente (formato HNSW) — apenas N e labels são lidos
//   - treePath:  vptree.bin — duas VP-Trees (seeds 1 e 2) + vetores int16
//
// Formato vptree.bin (dual-tree):
//
//	[4] uint32 N
//	[4] uint32 nodeCount1
//	[4] uint32 leafSize
//	[4] uint32 nodeCount2   ← identifica formato dual (>0)
//	[nodeCount1×16] vpNode, [N×4] perm1
//	[nodeCount2×16] vpNode, [N×4] perm2
//	[N×dim×2] int16
//
// Ambos mapeados via mmap(2) — zero cópia, zero pressão no GC.
// RSS de index.bin: apenas 16B header + N labels ≈ 3MB (vs 122MB total).
func LoadVP(indexPath, treePath string) (*VPIndex, error) {
	start := time.Now()
	log.Println("carregando VP-Tree via mmap...")

	idxData, err := mmapReadOnly(indexPath)
	if err != nil {
		return nil, fmt.Errorf("LoadVP index: %w", err)
	}
	treeData, err := mmapReadOnly(treePath)
	if err != nil {
		return nil, fmt.Errorf("LoadVP tree: %w", err)
	}

	// ── Parse index.bin: apenas header (N) e labels ──────────────────────────
	if len(idxData) < 16 {
		return nil, fmt.Errorf("index.bin muito pequeno: %d bytes", len(idxData))
	}
	N := int(binary.LittleEndian.Uint32(idxData[0:]))

	if need := 16 + N*dim + N; len(idxData) < need {
		return nil, fmt.Errorf("index.bin: esperado ≥%d bytes, tem %d", need, len(idxData))
	}
	labels := idxData[16+N*dim : 16+N*dim+N]

	// ── Parse vptree.bin (formato dual-tree) ─────────────────────────────────
	if len(treeData) < 16 {
		return nil, fmt.Errorf("vptree.bin muito pequeno: %d bytes", len(treeData))
	}
	treeN := int(binary.LittleEndian.Uint32(treeData[0:]))
	nodeCount1 := int(binary.LittleEndian.Uint32(treeData[4:]))
	// treeData[8:12] = leafSize (ignorado em runtime)
	nodeCount2 := int(binary.LittleEndian.Uint32(treeData[12:]))
	if treeN != N {
		return nil, fmt.Errorf("mismatch: index.bin N=%d, vptree.bin N=%d", N, treeN)
	}

	// Layout: 16B header | nc1×16B nodes1 | N×4B perm1 | nc2×16B nodes2 | N×4B perm2 | N×dim×2B vecs16
	off1 := 16
	offPerm1 := off1 + nodeCount1*16
	off2 := offPerm1 + N*4
	offPerm2 := off2 + nodeCount2*16
	offVecs16 := offPerm2 + N*4

	need := offVecs16 + N*dim*2
	if len(treeData) < need {
		return nil, fmt.Errorf("vptree.bin: esperado ≥%d bytes, tem %d", need, len(treeData))
	}

	nodes1 := unsafe.Slice((*vpNode)(unsafe.Pointer(&treeData[off1])), nodeCount1)
	perm1 := unsafe.Slice((*uint32)(unsafe.Pointer(&treeData[offPerm1])), N)
	nodes2 := unsafe.Slice((*vpNode)(unsafe.Pointer(&treeData[off2])), nodeCount2)
	perm2 := unsafe.Slice((*uint32)(unsafe.Pointer(&treeData[offPerm2])), N)
	vectors16 := unsafe.Slice((*int16)(unsafe.Pointer(&treeData[offVecs16])), N*dim)

	const poolSz = 2
	knnPool := make(chan *vpKNN, poolSz)
	for i := 0; i < poolSz; i++ {
		knnPool <- &vpKNN{}
	}
	// PQ pré-alocado: evita realocações durante busca refinada.
	// Com max_visits=700 e depth=16: até 11200 entradas por árvore.
	pqPool := make(chan *vpPQ, poolSz)
	pq2Pool := make(chan *vpPQ, poolSz)
	for i := 0; i < poolSz; i++ {
		pqPool <- &vpPQ{buf: make([]vpPQEntry, 0, vpPQInitialCap)}
		pq2Pool <- &vpPQ{buf: make([]vpPQEntry, 0, vpPQInitialCap)}
	}

	log.Printf("VP-Tree dual carregada: N=%d, %d+%d nós em %s", N, nodeCount1, nodeCount2, time.Since(start))
	return &VPIndex{
		n: N, vectors16: vectors16, labels: labels,
		nodes: nodes1, perm: perm1,
		nodes2: nodes2, perm2: perm2,
		knnPool: knnPool, pqPool: pqPool, pq2Pool: pq2Pool,
		mmapIdx: idxData, mmapTree: treeData,
	}, nil
}

// distSqF32 calcula a distância L2 AO QUADRADO (sem sqrt) entre a query float32
// e o vetor int16 do ponto ptIdx. Usado para TODOS os pontos (folha e VP).
// Para VPs de nós internos, o chamador aplica sqrt para o pruning.
//
// Reconstrução: ref_float = int16_val / 32767.0
// Sentinela: query[i]<0 ou ref[i]<0 → ambos: 0; só um: 1.0.
// Precisão: ±0.000015 por dimensão (vs ±0.004 do int8 — 250× mais preciso).
func (idx *VPIndex) distSqF32(query []float32, ptIdx uint32) float32 {
	ref := idx.vectors16[int(ptIdx)*dim : int(ptIdx)*dim+dim]
	const scale = float32(1.0 / 32767.0)
	var sum float32
	for i := 0; i < dim; i++ {
		qi := query[i]
		ri := ref[i]
		if qi < 0 || ri < 0 {
			if !(qi < 0 && ri < 0) {
				sum += 1.0
			}
			continue
		}
		d := qi - float32(ri)*scale
		sum += d * d
	}
	return sum
}

// ── vpKNN: max-heap fixo k=5, distâncias AO QUADRADO ─────────────────────────

func (h *vpKNN) reset() { h.n = 0 }

// worstSq retorna o quadrado da distância do pior vizinho atual.
// Retorna +∞ enquanto não está cheio (n < 5).
func (h *vpKNN) worstSq() float32 {
	if h.n < 5 {
		return float32(math.MaxFloat32)
	}
	m := h.dSq[0]
	for i := 1; i < 5; i++ {
		if h.dSq[i] > m {
			m = h.dSq[i]
		}
	}
	return m
}

// worstActual retorna a distância REAL do pior vizinho (sqrt do worstSq).
// Chamado poucas vezes por query (~20 vezes no pruning de nós internos).
func (h *vpKNN) worstActual() float32 {
	if h.n < 5 {
		return float32(math.MaxFloat32)
	}
	return float32(math.Sqrt(float64(h.worstSq())))
}

// tryInsertSq tenta inserir (dSq, pt) no heap de distâncias ao quadrado.
// Deduplicação: ignora o ponto se já está no heap (necessário para dual-tree,
// onde o mesmo ponto pode aparecer como VP em tree 1 e leaf em tree 2).
// Custo de dedup: O(k) = O(5) comparações por inserção — negligível.
func (h *vpKNN) tryInsertSq(dSq float32, pt uint32) {
	// Dedup: ponto já presente → ignora (evita duplicatas no fraud count)
	for i := 0; i < h.n; i++ {
		if h.idx[i] == pt {
			return
		}
	}
	if h.n < 5 {
		h.dSq[h.n] = dSq
		h.idx[h.n] = pt
		h.n++
		return
	}
	wi, wd := 0, h.dSq[0]
	for i := 1; i < 5; i++ {
		if h.dSq[i] > wd {
			wi, wd = i, h.dSq[i]
		}
	}
	if dSq < wd {
		h.dSq[wi] = dSq
		h.idx[wi] = pt
	}
}

// fraudCount conta quantos dos h.n vizinhos são fraude.
func (h *vpKNN) fraudCount(labels []uint8) int {
	count := 0
	for i := 0; i < h.n; i++ {
		if labels[h.idx[i]] == 1 {
			count++
		}
	}
	return count
}

// fraudFraction retorna a fração de fraudes em [0.0, 1.0].
func (h *vpKNN) fraudFraction(labels []uint8, k int) float32 {
	return float32(h.fraudCount(labels)) / float32(k)
}

// ── vpPQ: min-heap de backtracking (minDist = distância REAL) ────────────────

func (pq *vpPQ) push(e vpPQEntry) {
	pq.buf = append(pq.buf, e)
	i := len(pq.buf) - 1
	for i > 0 {
		p := (i - 1) >> 1
		if pq.buf[p].minDist <= pq.buf[i].minDist {
			break
		}
		pq.buf[p], pq.buf[i] = pq.buf[i], pq.buf[p]
		i = p
	}
}

func (pq *vpPQ) pop() vpPQEntry {
	n := len(pq.buf) - 1
	pq.buf[0], pq.buf[n] = pq.buf[n], pq.buf[0]
	e := pq.buf[n]
	pq.buf = pq.buf[:n]
	i := 0
	for {
		l := 2*i + 1
		if l >= len(pq.buf) {
			break
		}
		j := l
		if r := l + 1; r < len(pq.buf) && pq.buf[r].minDist < pq.buf[l].minDist {
			j = r
		}
		if pq.buf[i].minDist <= pq.buf[j].minDist {
			break
		}
		pq.buf[i], pq.buf[j] = pq.buf[j], pq.buf[i]
		i = j
	}
	return e
}

// ── greedyToLeaf: desce greedy da raiz até a folha mais próxima ──────────────

// greedyToLeaf desce greedily de nodeIdx até a primeira folha, escolhendo sempre
// o ramo mais próximo. Os ramos descartados são salvos em pq para backtracking.
// Preenche knn com o VP de cada nó interno e todos os pontos da folha.
// nodes/perm identificam qual das duas árvores usar (árvore 1 ou 2).
func (idx *VPIndex) greedyToLeaf(nodes []vpNode, perm []uint32, query []float32, nodeIdx int32, knn *vpKNN, pq *vpPQ) {
	for {
		node := &nodes[nodeIdx]
		if node.left < 0 {
			// ── Folha: avalia todos os pontos (sem sqrt) ─────────────────
			lo := int(-(node.left + 1))
			hi := int(-(node.right + 1))
			for i := lo; i < hi; i++ {
				knn.tryInsertSq(idx.distSqF32(query, perm[i]), perm[i])
			}
			return
		}

		// ── Nó interno: 1 sqrt para a distância do VP ────────────────────
		dSq := idx.distSqF32(query, node.vpIdx)
		d := float32(math.Sqrt(float64(dSq)))
		knn.tryInsertSq(dSq, node.vpIdx)

		μ := node.medDist
		worstAct := knn.worstActual()

		if d <= μ {
			// Query dentro da bola: left é o mais próximo; salva right
			if knn.n < 5 || μ-d < worstAct {
				pq.push(vpPQEntry{node.right, μ - d})
			}
			nodeIdx = node.left
		} else {
			// Query fora da bola: right é o mais próximo; salva left
			if knn.n < 5 || d-μ < worstAct {
				pq.push(vpPQEntry{node.left, d - μ})
			}
			nodeIdx = node.right
		}
	}
}

// Search encontra os k vizinhos mais próximos de query e retorna a fração
// de vizinhos fraudulentos (fraud score em [0.0, 1.0]).
//
// Algoritmo: greedy descent single-tree + backtracking em dois estágios.
//
//  1. Estágio rápido (todos os casos): vpInitialLeafVisits descents (tree 1).
//     Custo: ~5×80 ops × 7ns ≈ 2.8µs.
//     Cobre 96%+ dos casos (fraudCount 0 ou k = decisão clara).
//
//  2. Refinamento (só casos borderline, fraudCount ∈ {1..k-1}):
//     Continua explorando via PQ de backtracking da tree 1 até vpMaxLeafVisits.
//     Custo: até ~700×80 ops × 7ns ≈ 392µs (para ~4% dos casos).
//
// Custo médio (N=3M, 96% clear + 4% borderline):
//
//	0.96×2.8 + 0.04×392 ≈ 18.4µs/query média.
//
// Nota: pq2 é adquirido do pool para manter compatibilidade com o formato
// dual-tree do vptree.bin, mas não é usado na busca (tree 1 é suficiente).
func (idx *VPIndex) Search(query []float32, k int) float32 {
	knn := <-idx.knnPool
	pq := <-idx.pqPool
	pq2 := <-idx.pq2Pool // adquirido mas não usado na busca
	defer func() {
		knn.reset()
		idx.knnPool <- knn
	}()
	defer func() {
		pq.buf = pq.buf[:0]
		idx.pqPool <- pq
	}()
	defer func() {
		pq2.buf = pq2.buf[:0]
		idx.pq2Pool <- pq2
	}()

	// ── Estágio 1: vpInitialLeafVisits descents greedy (tree 1) ──────────
	idx.greedyToLeaf(idx.nodes, idx.perm, query, 0, knn, pq)
	for leafVisits := 1; leafVisits < vpInitialLeafVisits && len(pq.buf) > 0; leafVisits++ {
		c := pq.pop()
		if knn.n == k && c.minDist >= knn.worstActual() {
			break
		}
		idx.greedyToLeaf(idx.nodes, idx.perm, query, c.nodeIdx, knn, pq)
	}

	// ── Estágio 2: refinamento só para casos borderline (tree 1) ─────────
	// Casos borderline (fraudCount ∈ {1..k-1}) recebem busca extra via PQ.
	// Stage 1 já preencheu o PQ com nós a explorar — continuamos de onde paramos.
	fc := knn.fraudCount(idx.labels)
	if fc > 0 && fc < k {
		for leafVisits := vpInitialLeafVisits; leafVisits < vpMaxLeafVisits && len(pq.buf) > 0; leafVisits++ {
			c := pq.pop()
			if knn.n == k && c.minDist >= knn.worstActual() {
				break
			}
			idx.greedyToLeaf(idx.nodes, idx.perm, query, c.nodeIdx, knn, pq)
		}
	}

	return knn.fraudFraction(idx.labels, k)
}

// WarmUp pré-carrega páginas de vptree.bin no page cache do OS via buscas reais.
//
// O problema: vptree.bin (~107MB) é mmap'd mas as páginas só entram no cache
// quando acessadas. Com cold start, as primeiras N requisições sofrem page faults
// (disk I/O latência ≈ 0.1–1 ms cada) que aumentam o p99 no início do teste.
//
// Solução: fazer nWarming buscas greedy usando vetores reais do índice como query,
// dispersados uniformemente pelo array. Cada busca acessa ~80 pontos e atravessa
// depth=16 nós internos, cobrindo ≈ 2.5 kB de páginas distintas.
// Com 3000 buscas: ~7.5 MB de páginas de nodes+vectors acessadas aleatoriamente —
// suficiente para precarregar os nós raiz (L0–L11) e o path mais quente da árvore.
//
// Nota: não faz leitura sequencial de vectors16 (84 MB) para evitar exceder
// o limite de 170 MB de memória do container.
func (idx *VPIndex) WarmUp() {
	const nWarming = 3000
	knn := &vpKNN{}
	pq := &vpPQ{buf: make([]vpPQEntry, 0, 64)}

	step := idx.n / nWarming
	if step < 1 {
		step = 1
	}

	for i := 0; i < nWarming && i*step < idx.n; i++ {
		ptIdx := i * step
		// Constrói query float32 a partir do vetor int16 do ponto ptIdx
		var query [dim]float32
		ref := idx.vectors16[ptIdx*dim : ptIdx*dim+dim]
		const scale = float32(1.0 / 32767.0)
		for j := 0; j < dim; j++ {
			if ref[j] < 0 {
				query[j] = -1.0
			} else {
				query[j] = float32(ref[j]) * scale
			}
		}
		knn.reset()
		pq.buf = pq.buf[:0]
		idx.greedyToLeaf(idx.nodes, idx.perm, query[:], 0, knn, pq)
	}
}
