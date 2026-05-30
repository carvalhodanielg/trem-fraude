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
// Precisão: float32 (sem quantização na query — só na referência int8).
// Poda: triangle inequality usada para cada "outro filho" salvo no PQ.
// Early exit: se após a 1ª folha o resultado já é 0/k ou k/k, retorna imediato.
//
// Formato dos arquivos:
//
//	index.bin: mesmo formato HNSW (reutilizado para vetores+labels)
//	vptree.bin: header(16B) + nodes(nodeCount×16B) + perm(N×4B)
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

// vpMaxLeafVisits é o número máximo de folhas visitadas por query.
// Custo por folha ≈ depth_parcial × (1 dist+sqrt) + leafSize × dist_sq.
// vpMaxLeafVisits=5, leafSize=64: ~5×(~4+64) ≈ 340 ops × 7ns ≈ 2.5µs/query.
// Aumente para mais precisão (edge cases); reduza para menos latência.
const vpMaxLeafVisits = 5

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
type VPIndex struct {
	n       int
	vectors []int8   // mmap'd de index.bin: N×dim int8 quantizados
	labels  []uint8  // mmap'd de index.bin: N uint8 (0=legit, 1=fraud)
	nodes   []vpNode // mmap'd de vptree.bin: nós da VP-Tree
	perm    []uint32 // mmap'd de vptree.bin: índices originais na ordem VP-Tree

	// Pools de tamanho 2: com GOMAXPROCS=1, 1 em uso + 1 de folga.
	knnPool chan *vpKNN
	pqPool  chan *vpPQ

	mmapIdx  []byte // mantém index.bin vivo (mmap)
	mmapTree []byte // mantém vptree.bin vivo (mmap)
}

// LoadVP carrega o índice VP-Tree a partir de dois arquivos:
//   - indexPath: index.bin existente (formato HNSW) — apenas vectors+labels são lidos
//   - treePath:  vptree.bin — estrutura da VP-Tree (nodes + perm)
//
// Ambos mapeados via mmap(2) — zero cópia, zero pressão no GC.
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

	// ── Parse index.bin (header: N, M, nLayers, entryIdx) ───────────────────
	if len(idxData) < 16 {
		return nil, fmt.Errorf("index.bin muito pequeno: %d bytes", len(idxData))
	}
	N := int(binary.LittleEndian.Uint32(idxData[0:]))
	// M, nLayers, entryIdx não usados pelo VP-Tree

	if need := 16 + N*dim + N; len(idxData) < need {
		return nil, fmt.Errorf("index.bin: esperado ≥%d bytes, tem %d", need, len(idxData))
	}
	vectors := unsafe.Slice((*int8)(unsafe.Pointer(&idxData[16])), N*dim)
	labels := idxData[16+N*dim : 16+N*dim+N]

	// ── Parse vptree.bin (header: N, nodeCount, leafSize, pad) ──────────────
	if len(treeData) < 16 {
		return nil, fmt.Errorf("vptree.bin muito pequeno: %d bytes", len(treeData))
	}
	treeN := int(binary.LittleEndian.Uint32(treeData[0:]))
	nodeCount := int(binary.LittleEndian.Uint32(treeData[4:]))
	if treeN != N {
		return nil, fmt.Errorf("mismatch: index.bin N=%d, vptree.bin N=%d", N, treeN)
	}
	if need := 16 + nodeCount*16 + N*4; len(treeData) < need {
		return nil, fmt.Errorf("vptree.bin: esperado ≥%d bytes, tem %d", need, len(treeData))
	}
	// Alinhamento 4: offset 16 é múltiplo de 4 (sizeof vpNode = 16). ✓
	nodes := unsafe.Slice((*vpNode)(unsafe.Pointer(&treeData[16])), nodeCount)
	perm := unsafe.Slice((*uint32)(unsafe.Pointer(&treeData[16+nodeCount*16])), N)

	const poolSz = 2
	knnPool := make(chan *vpKNN, poolSz)
	for i := 0; i < poolSz; i++ {
		knnPool <- &vpKNN{}
	}
	// PQ depth = tree depth ≤ 20 entradas; pre-aloca 64 para folga.
	pqPool := make(chan *vpPQ, poolSz)
	for i := 0; i < poolSz; i++ {
		pqPool <- &vpPQ{buf: make([]vpPQEntry, 0, 64)}
	}

	log.Printf("VP-Tree carregada: N=%d, %d nós em %s", N, nodeCount, time.Since(start))
	return &VPIndex{
		n: N, vectors: vectors, labels: labels,
		nodes: nodes, perm: perm,
		knnPool: knnPool, pqPool: pqPool,
		mmapIdx: idxData, mmapTree: treeData,
	}, nil
}

// distSqF32 calcula a distância L2 AO QUADRADO (sem sqrt) entre a query float32
// e o vetor int8 do ponto ptIdx. Usado para TODOS os pontos (folha e VP).
// Para VPs de nós internos, o chamador aplica sqrt para o pruning.
//
// Reconstrução: ref_float = int8_val / 127.0
// Sentinela: query[i]<0 ou ref[i]==-1 → ambos: 0; só um: 1.0.
func (idx *VPIndex) distSqF32(query []float32, ptIdx uint32) float32 {
	ref := idx.vectors[int(ptIdx)*dim : int(ptIdx)*dim+dim]
	const scale = float32(1.0 / 127.0)
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
func (h *vpKNN) tryInsertSq(dSq float32, pt uint32) {
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
func (idx *VPIndex) greedyToLeaf(query []float32, nodeIdx int32, knn *vpKNN, pq *vpPQ) {
	for {
		node := &idx.nodes[nodeIdx]
		if node.left < 0 {
			// ── Folha: avalia todos os pontos (sem sqrt) ─────────────────
			lo := int(-(node.left + 1))
			hi := int(-(node.right + 1))
			for i := lo; i < hi; i++ {
				knn.tryInsertSq(idx.distSqF32(query, idx.perm[i]), idx.perm[i])
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
// Algoritmo: greedy descent + backtracking limitado.
//
//  1. Desce greedy da raiz até a folha mais próxima (O(depth) = ~16 passos).
//     Salva os ramos descartados no PQ de backtracking com seus minDists.
//  2. Early exit: se após a 1ª folha fraudCount=0 ou k, retorna imediato.
//  3. Backtracking: explora até vpMaxLeafVisits-1 folhas adicionais,
//     priorizando pelo minDist (triângulo). Para cada entrada do PQ,
//     desce greedy até a próxima folha.
//
// Custo por query (N=3M, leafSize=64, vpMaxLeafVisits=5):
//
//	~16 internos (16 sqrts) + 5×64 leaf (0 sqrts) ≈ 340 ops × 7ns ≈ 2.4µs
//
// Precisão: float32 para queries; int8 reconstruído para referências.
// Melhor que HNSW int8 para edge cases (fraudCount ∈ {1,2,3,4}).
func (idx *VPIndex) Search(query []float32, k int) float32 {
	knn := <-idx.knnPool
	pq := <-idx.pqPool
	defer func() {
		knn.reset()
		idx.knnPool <- knn
	}()
	defer func() {
		pq.buf = pq.buf[:0]
		idx.pqPool <- pq
	}()

	// ── Fase 1: greedy descent até a primeira folha ───────────────────────
	idx.greedyToLeaf(query, 0, knn, pq)

	// ── Fase 2: backtracking — explora folhas adicionais ─────────────────
	// Sempre explora vpMaxLeafVisits folhas para precisão máxima.
	// O early exit HNSW-style não é aplicado aqui: com leafSize=64 a
	// primeira folha dá apenas ~80 candidatos — pode ser prematuro para
	// casos borderline (fraudCount ∈ {1,2,3,4} que pesam 3× no score).
	for leafVisits := 1; leafVisits < vpMaxLeafVisits && len(pq.buf) > 0; leafVisits++ {
		c := pq.pop()
		if knn.n == k && c.minDist >= knn.worstActual() {
			break // prune por triangle inequality
		}
		idx.greedyToLeaf(query, c.nodeIdx, knn, pq)
	}

	return knn.fraudFraction(idx.labels, k)
}
