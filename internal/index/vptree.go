// Package index — VP-Tree (Vantage Point Tree) para busca k-NN exata.
//
// Substitui o HNSW (busca aproximada) por busca exata dentro do erro de
// quantização int8. O objetivo é reduzir failure_rate para próximo de 0%
// eliminando erros de aproximação do HNSW.
//
// Formato dos arquivos:
//
//	index.bin: igual ao formato HNSW existente (reutilizado para vetores+labels)
//	vptree.bin: estrutura da VP-Tree (header + nodes + perm)
//
// Estrutura da VP-Tree:
//   - Nó interno: vantage point + medDist + filhos esq/dir
//   - Nó folha: range em perm[] codificado como valores negativos
//   - left < 0 → folha com lo = -(left+1), hi = -(right+1)
//   - left ≥ 0 → nó interno com filhos em nodes[left] e nodes[right]
//
// Algoritmo de busca:
//   Priority queue (min-heap) de subárvores a visitar, ordenada por
//   distância mínima possível. Poda por triangle inequality:
//     - Dentro da bola (left):  minDist = max(0, d − μ)
//     - Fora da bola  (right): minDist = max(0, μ − d)
//   onde d = dist(query, vp) e μ = medDist do nó.
package index

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"time"
	"unsafe"
)

// vpNode é um nó da VP-Tree, armazenado em vptree.bin (16 bytes cada).
//
// Nó interno: vpIdx válido, medDist válido, left ≥ 0, right ≥ 0.
// Nó folha:   left = -(lo+1) < 0, right = -(hi+1) < 0.
//
//	Recuperação da folha: lo = -(left+1), hi = -(right+1)
type vpNode struct {
	vpIdx   uint32  // índice do vantage point nos arrays vectors/labels
	medDist float32 // distância mediana para partição da bola
	left    int32   // filho esquerdo (≥0) ou -(lo+1) se folha
	right   int32   // filho direito  (≥0) ou -(hi+1) se folha
}

// vpKNN é um max-heap fixo para os k=5 vizinhos mais próximos encontrados.
// Evita alocações no hot path — reutilizado via pool.
type vpKNN struct {
	d   [5]float32 // distâncias (mantidas em ordem arbitrária)
	idx [5]uint32  // índices originais dos pontos
	n   int        // quantidade preenchida (0 ≤ n ≤ 5)
}

// vpPQEntry é uma entrada na priority queue da busca.
type vpPQEntry struct {
	nodeIdx int32   // índice em VPIndex.nodes
	minDist float32 // limite inferior da distância a qualquer ponto nessa subárvore
}

// vpPQ é uma min-heap de vpPQEntry, ordenada por minDist.
type vpPQ struct {
	buf []vpPQEntry
}

// VPIndex é o índice VP-Tree carregado via mmap.
type VPIndex struct {
	n       int
	vectors []int8  // mmap'd de index.bin: N×dim int8 quantizados
	labels  []uint8 // mmap'd de index.bin: N uint8 (0=legit, 1=fraud)
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
// Ambos os arquivos são mapeados via mmap(2) — zero cópia, zero pressão no GC.
// Os dois containers Docker compartilham as mesmas páginas físicas via overlayfs.
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

	// ── Parse index.bin header ───────────────────────────────────────────────
	// Formato: uint32 N, uint32 M, uint32 nLayers, uint32 entryIdx, then vectors+labels.
	if len(idxData) < 16 {
		return nil, fmt.Errorf("index.bin muito pequeno: %d bytes", len(idxData))
	}
	N := int(binary.LittleEndian.Uint32(idxData[0:]))
	// M, nLayers, entryIdx não usados pelo VP-Tree; ignoramos.

	need := 16 + N*dim + N
	if len(idxData) < need {
		return nil, fmt.Errorf("index.bin: esperado ≥%d bytes, tem %d", need, len(idxData))
	}

	vectors := unsafe.Slice((*int8)(unsafe.Pointer(&idxData[16])), N*dim)
	labels := idxData[16+N*dim : 16+N*dim+N]

	// ── Parse vptree.bin header ──────────────────────────────────────────────
	// Formato: uint32 N, uint32 nodeCount, uint32 leafSize, uint32 pad, então nodes+perm.
	if len(treeData) < 16 {
		return nil, fmt.Errorf("vptree.bin muito pequeno: %d bytes", len(treeData))
	}
	treeN := int(binary.LittleEndian.Uint32(treeData[0:]))
	nodeCount := int(binary.LittleEndian.Uint32(treeData[4:]))
	// leafSize e pad ignorados em runtime

	if treeN != N {
		return nil, fmt.Errorf("mismatch: index.bin N=%d, vptree.bin N=%d", N, treeN)
	}

	treeNeed := 16 + nodeCount*16 + N*4
	if len(treeData) < treeNeed {
		return nil, fmt.Errorf("vptree.bin: esperado ≥%d bytes, tem %d", treeNeed, len(treeData))
	}

	// Alinhamento 4: offset 16 é múltiplo de 4 (sizeof vpNode = 16). ✓
	nodes := unsafe.Slice((*vpNode)(unsafe.Pointer(&treeData[16])), nodeCount)
	perm := unsafe.Slice((*uint32)(unsafe.Pointer(&treeData[16+nodeCount*16])), N)

	// ── Pools ────────────────────────────────────────────────────────────────
	const poolSz = 2
	knnPool := make(chan *vpKNN, poolSz)
	for i := 0; i < poolSz; i++ {
		knnPool <- &vpKNN{}
	}
	pqPool := make(chan *vpPQ, poolSz)
	for i := 0; i < poolSz; i++ {
		pqPool <- &vpPQ{buf: make([]vpPQEntry, 0, 512)}
	}

	log.Printf("VP-Tree carregada: N=%d, %d nós em %s", N, nodeCount, time.Since(start))

	return &VPIndex{
		n:        N,
		vectors:  vectors,
		labels:   labels,
		nodes:    nodes,
		perm:     perm,
		knnPool:  knnPool,
		pqPool:   pqPool,
		mmapIdx:  idxData,
		mmapTree: treeData,
	}, nil
}

// distFloat calcula a distância euclidiana real (não ao quadrado) entre
// a query float32 e o vetor de referência int8 do ponto ptIdx.
//
// Reconstrução: ref_float = int8_val / 127.0
// Sentinela: query[i] < 0 ou ref[i] == -1 → se ambos: contribui 0;
//
//	se só um: contribui 1.0 (penalidade máxima normalizada).
func (idx *VPIndex) distFloat(query []float32, ptIdx uint32) float32 {
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
	return float32(math.Sqrt(float64(sum)))
}

// ── vpKNN: max-heap fixo k=5 ─────────────────────────────────────────────────

func (h *vpKNN) reset() { h.n = 0 }

// worst retorna a distância do pior candidato atual.
// Retorna +∞ enquanto h não está cheio (n < 5).
func (h *vpKNN) worst() float32 {
	if h.n < 5 {
		return float32(math.MaxFloat32)
	}
	m := h.d[0]
	for i := 1; i < 5; i++ {
		if h.d[i] > m {
			m = h.d[i]
		}
	}
	return m
}

// tryInsert tenta inserir (d, pt) no heap.
// Se não está cheio: insere. Se cheio: substitui o pior se d < pior.
func (h *vpKNN) tryInsert(d float32, pt uint32) {
	if h.n < 5 {
		h.d[h.n] = d
		h.idx[h.n] = pt
		h.n++
		return
	}
	// Encontra o pior candidato atual
	wi, wd := 0, h.d[0]
	for i := 1; i < 5; i++ {
		if h.d[i] > wd {
			wi, wd = i, h.d[i]
		}
	}
	if d < wd {
		h.d[wi] = d
		h.idx[wi] = pt
	}
}

// fraudFraction conta fraudes no top-k e retorna a fração em [0.0, 1.0].
func (h *vpKNN) fraudFraction(labels []uint8, k int) float32 {
	count := 0
	for i := 0; i < h.n; i++ {
		if labels[h.idx[i]] == 1 {
			count++
		}
	}
	return float32(count) / float32(k)
}

// ── vpPQ: min-heap de subárvores a visitar ───────────────────────────────────

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
	// sift down
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

// Search encontra os k vizinhos mais próximos de query e retorna a fração
// de vizinhos fraudulentos (fraud score em [0.0, 1.0]).
//
// Algoritmo: priority queue sobre subárvores, podando por triangle inequality.
// Busca exata (dentro do erro de quantização int8) — sem aproximação HNSW.
//
// Distância: L2 euclidiana real (com sqrt) para poda correta pelo triangulo.
// Os vetores de referência são int8 reconstruídos para float32.
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

	// Começa pela raiz (nodeIdx=0, minDist=0)
	pq.push(vpPQEntry{0, 0.0})

	for len(pq.buf) > 0 {
		c := pq.pop()

		// Poda: se a menor distância possível nesta subárvore já é ≥ ao pior
		// candidato atual no knn heap, todo o restante da PQ também é pior.
		// (A PQ é min-heap; se o topo já é inútil, todos são.)
		if knn.n == k && c.minDist >= knn.worst() {
			break
		}

		node := &idx.nodes[c.nodeIdx]

		if node.left < 0 {
			// ── Folha: verifica todos os pontos no range perm[lo:hi] ──────
			lo := int(-(node.left + 1))
			hi := int(-(node.right + 1))
			for i := lo; i < hi; i++ {
				pt := idx.perm[i]
				d := idx.distFloat(query, pt)
				knn.tryInsert(d, pt)
			}
		} else {
			// ── Nó interno: verifica VP e empurra os dois filhos ──────────
			d := idx.distFloat(query, node.vpIdx)
			knn.tryInsert(d, node.vpIdx)

			μ := node.medDist

			// Triangle inequality bounds:
			//   Dentro (left):  todos x têm dist(vp,x) ≤ μ
			//     → dist(q,x) ≥ max(0, d−μ)
			//   Fora  (right): todos x têm dist(vp,x) ≥ μ
			//     → dist(q,x) ≥ max(0, μ−d)
			var minL, minR float32
			if d <= μ {
				minL = 0     // query dentro da bola → left está próximo
				minR = μ - d // right está pelo menos μ−d longe
			} else {
				minL = d - μ // left está pelo menos d−μ longe
				minR = 0     // query fora da bola → right está próximo
			}

			// Empurra filhos; poda imediata se já sabemos que são inúteis.
			worst := knn.worst()
			if knn.n < k || minL < worst {
				pq.push(vpPQEntry{node.left, minL})
			}
			if knn.n < k || minR < worst {
				pq.push(vpPQEntry{node.right, minR})
			}
		}
	}

	return knn.fraudFraction(idx.labels, k)
}
