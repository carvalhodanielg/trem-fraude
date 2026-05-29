// Package index implementa busca HNSW sobre o índice compacto gerado
// pelo cmd/preprocess. O índice usa vetores quantizados em int8 e
// adjacência em arrays flat (sem maps).
// O Load usa mmap(2) para mapear o arquivo diretamente — zero cópia,
// sem pressão no heap Go nem no GC. Dois containers da mesma imagem Docker
// (overlayfs) compartilham as mesmas physical pages do page cache.
package index

import (
	"encoding/binary"
	"log"
	"time"
	"unsafe"
)

const (
	dim      = 14
	efSearch = 60 // candidatos na busca do nível 0 (stage 2)
	efFast   = 30 // busca rápida para casos claros (stage 1)
)

// visitedBitset é um bitset reutilizável para rastrear nós visitados no beam search.
// Usa []uint64 em vez de map para evitar cache misses e pressão de GC.
// O campo toReset rastreia quais bits foram marcados, permitindo limpeza eficiente
// (O(nós visitados) em vez de O(N/64) de um memset completo de 375 KB).
type visitedBitset struct {
	bits    []uint64
	toReset []int
}

// heapBufs mantém os slices dos dois heaps de beam search pré-alocados.
// Reutilizar esses slices entre requisições elimina alocações no hot path e
// reduz pressão de GC — crítico com GOMAXPROCS=1.
// cands: candidatos a explorar (minHeap, cresce até ~stride*efSearch elementos).
// res:   melhores resultados encontrados (maxHeap, cresce até efSearch elementos).
type heapBufs struct {
	cands []cand
	res   []cand
}

// Index é o índice HNSW carregado em memória.
type Index struct {
	n, m, nLayers, entry int

	// Dados do índice: views zero-cópia sobre o arquivo mmap'd (mmapData).
	// O GC Go não gerencia nem escaneia o backing array desses slices.
	vectors []int8       // flat, stride dim: nó i → vectors[i*dim:(i+1)*dim]
	labels  []uint8      // labels[i] = 0 (legit) ou 1 (fraud)
	adj0    []byte       // flat, 3 bytes/vizinho (uint24 LE): nó i → adj0[i*stride*3:(i*stride+adj0Cnt[i])*3]
	adj0Cnt []uint8      // adj0Cnt[i] = vizinhos válidos do nó i na camada 0
	upper   []upperLayer // upper[l] = camada l+1 do grafo HNSW

	// visited é um canal de bitsets pré-alocados usados no beam search.
	// Com GOMAXPROCS=1, tamanho 2 é suficiente: 1 em uso + 1 de folga para
	// preempção assíncrona do scheduler. O canal mantém os bitsets vivos
	// (evita GC) e fornece backpressure sem deadlock.
	visited chan *visitedBitset

	// heaps é um canal de heapBufs pré-alocados, mesmo padrão que visited.
	// Elimina alocações de minCandHeap e maxCandHeap a cada requisição.
	heaps chan *heapBufs

	// mmapData mantém o slice mmap'd vivo enquanto o Index existir.
	// Todos os slices de dados acima são views deste slice.
	mmapData []byte
}

// upperLayer armazena a adjacência de uma camada superior (l >= 1) em CSR.
type upperLayer struct {
	nodes []int32  // nodeIdxs ordenados crescente (Nl entradas)
	off   []uint32 // offsets CSR (Nl+1 entradas)
	nbr   []uint32 // vizinhos (off[Nl] entradas)
}

// Load lê o binário compacto gerado pelo cmd/preprocess via mmap(2).
//
// Todos os arrays de dados (vectors, labels, adj0, adj0Cnt, upper CSR) são views
// zero-cópia do arquivo mapeado — não pressionam o heap Go nem o GC.
// Dois containers da mesma imagem Docker (overlayfs read-only layer) compartilham
// as mesmas physical pages do page cache do kernel: ~122 MB pagos uma vez no total.
//
// Alinhamento dos uint32/int32 nas camadas superiores garantido porque:
//   header(16) + vectors(N×14) + labels(N) + adj0(N×stride×3) + adj0Cnt(N)
//   = 16 + N×(14+1+18+1) = 16 + N×34
// Para N=3.000.000: 16 + 102.000.000 = 102.000.016 ≡ 0 (mod 4). ✓
func Load(path string) (*Index, error) {
	start := time.Now()
	log.Println("carregando índice HNSW via mmap...")

	data, err := mmapReadOnly(path)
	if err != nil {
		return nil, err
	}

	cur := 0
	readU32 := func() uint32 {
		v := binary.LittleEndian.Uint32(data[cur:])
		cur += 4
		return v
	}

	N := int(readU32())
	M := int(readU32())
	nLayers := int(readU32())
	entry := int(readU32())
	stride := 2 * M

	// Vetores int8 — view direta (alinhamento 1: sempre OK).
	vectors := unsafe.Slice((*int8)(unsafe.Pointer(&data[cur])), N*dim)
	cur += N * dim

	// Labels uint8 = byte — slice direto.
	labels := data[cur : cur+N]
	cur += N

	// Camada 0: adj0 uint24 — slice direto como []byte.
	adj0Size := N * stride * 3
	adj0 := data[cur : cur+adj0Size]
	cur += adj0Size

	// adj0Cnt uint8 — slice direto.
	adj0Cnt := data[cur : cur+N]
	cur += N

	// Camadas superiores (l=1..nLayers-1): CSR com views uint32/int32.
	// Alinhamento 4 verificado no comentário do Load acima.
	upper := make([]upperLayer, 0, nLayers-1)
	for l := 1; l < nLayers; l++ {
		Nl := int(readU32())

		nodes := unsafe.Slice((*int32)(unsafe.Pointer(&data[cur])), Nl)
		cur += Nl * 4

		off := unsafe.Slice((*uint32)(unsafe.Pointer(&data[cur])), Nl+1)
		cur += (Nl + 1) * 4

		nbrCount := int(off[Nl])
		nbr := unsafe.Slice((*uint32)(unsafe.Pointer(&data[cur])), nbrCount)
		cur += nbrCount * 4

		upper = append(upper, upperLayer{nodes: nodes, off: off, nbr: nbr})
	}

	log.Printf("índice HNSW carregado: %d nós, M=%d, %d camadas em %s",
		N, M, nLayers, time.Since(start))

	// Pool tamanho 2: com GOMAXPROCS=1, 1 bitset em uso + 1 folga para
	// preempção assíncrona do scheduler. Memória: 2×375KB ≈ 750 KB (vs 32×375KB≈12MB antes).
	bitWords := (N + 63) / 64
	const poolSize = 2
	visitedCh := make(chan *visitedBitset, poolSize)
	for i := 0; i < poolSize; i++ {
		visitedCh <- &visitedBitset{
			bits:    make([]uint64, bitWords),
			toReset: make([]int, 0, 512),
		}
	}

	// Pool de heapBufs tamanho 2 pelo mesmo motivo.
	// cands pode crescer até stride*efSearch = 6*60 = 360 elementos; res até efSearch=60.
	candsCap := stride*efSearch + 16
	const heapPoolSize = 2
	heapsCh := make(chan *heapBufs, heapPoolSize)
	for i := 0; i < heapPoolSize; i++ {
		heapsCh <- &heapBufs{
			cands: make([]cand, 0, candsCap),
			res:   make([]cand, 0, efSearch+1),
		}
	}

	return &Index{
		n: N, m: M, nLayers: nLayers, entry: entry,
		vectors:  vectors,
		labels:   labels,
		adj0:     adj0,
		adj0Cnt:  adj0Cnt,
		upper:    upper,
		visited:  visitedCh,
		heaps:    heapsCh,
		mmapData: data,
	}, nil
}

// vecAt retorna o slice de dim int8 do nó i.
func (idx *Index) vecAt(i int) []int8 {
	return idx.vectors[i*dim : (i+1)*dim]
}

// quantizeQuery converte o vetor de query float32 para [dim]int8.
func quantizeQuery(q []float32) [dim]int8 {
	var v [dim]int8
	for i, x := range q {
		if x < 0 {
			v[i] = -1 // sentinela
		} else {
			r := x * 127
			if r > 127 {
				r = 127
			}
			v[i] = int8(r + 0.5)
		}
	}
	return v
}

// distSq calcula a distância euclidiana ao quadrado em escala int8.
// Sentinela −1: se ambos são −1 → contribui 0; se só um → contribui 127².
func distSq(a *[dim]int8, b []int8) int32 {
	var sum int32
	for i := 0; i < dim; i++ {
		ai, bi := int32(a[i]), int32(b[i])
		if ai < 0 || bi < 0 {
			if ai != bi {
				sum += 127 * 127
			}
			continue
		}
		d := ai - bi
		sum += d * d
	}
	return sum
}

// distSqFloat calcula a distância euclidiana ao quadrado entre a query float32
// original e um vetor de referência int8 (promovido para float32).
// Usada no rerank pós-search para precisão máxima sem custo de memória extra.
// Sentinela: query[i] < 0 → ambos sentinela: contribui 0; só um: contribui 1.0
// (equivalente a 127/127 em escala normalizada, sem o 127² do int8).
func distSqFloat(query []float32, ref []int8) float32 {
	var sum float32
	const scale = 1.0 / 127.0
	for i := 0; i < dim; i++ {
		qi := query[i]
		ri := float32(ref[i]) * scale
		if qi < 0 || ref[i] < 0 {
			if !(qi < 0 && ref[i] < 0) {
				sum += 1.0 // penalidade máxima para sentinela misto
			}
			continue
		}
		d := qi - ri
		sum += d * d
	}
	return sum
}

// upperPos encontra a posição de nodeIdx em ul.nodes via busca binária.
// Retorna −1 se não encontrado.
func upperPos(ul *upperLayer, nodeIdx int) int {
	lo, hi := 0, len(ul.nodes)-1
	for lo <= hi {
		mid := (lo + hi) >> 1
		v := int(ul.nodes[mid])
		switch {
		case v == nodeIdx:
			return mid
		case v < nodeIdx:
			lo = mid + 1
		default:
			hi = mid - 1
		}
	}
	return -1
}

// greedyDescend desce pela camada superior ul a partir de ep,
// retornando o vizinho mais próximo de q nessa camada.
func (idx *Index) greedyDescend(ul *upperLayer, ep int, q *[dim]int8) int {
	best := ep
	bestD := distSq(q, idx.vecAt(ep))

	for {
		pos := upperPos(ul, best)
		if pos < 0 {
			break
		}
		improved := false
		start, end := ul.off[pos], ul.off[pos+1]
		for i := start; i < end; i++ {
			nbr := int(ul.nbr[i])
			d := distSq(q, idx.vecAt(nbr))
			if d < bestD {
				bestD = d
				best = nbr
				improved = true
			}
		}
		if !improved {
			break
		}
	}
	return best
}

// ── Heaps inline para o beam search (sem container/heap, sem boxing) ─────────
//
// Todas as operações são sobre []cand diretamente — sem interface{}, sem boxing,
// sem dispatch dinâmico. O compilador pode inlinar e vetorizar.
//
// minHeap: menor distância no topo (candidatos a explorar).
// maxHeap: maior distância no topo (janela dos ef melhores resultados).

type cand struct {
	idx int
	d   int32
}

// ── minHeap (candidatos) ──────────────────────────────────────────────────────

func minHeapUp(h []cand, i int) {
	for i > 0 {
		parent := (i - 1) >> 1
		if h[parent].d <= h[i].d {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
}

func minHeapDown(h []cand, i, n int) {
	for {
		left := (i << 1) + 1
		if left >= n {
			break
		}
		j := left
		if right := left + 1; right < n && h[right].d < h[left].d {
			j = right
		}
		if h[i].d <= h[j].d {
			break
		}
		h[i], h[j] = h[j], h[i]
		i = j
	}
}

func minHeapPush(h *[]cand, c cand) {
	*h = append(*h, c)
	minHeapUp(*h, len(*h)-1)
}

func minHeapPop(h *[]cand) cand {
	n := len(*h) - 1
	(*h)[0], (*h)[n] = (*h)[n], (*h)[0]
	minHeapDown(*h, 0, n)
	x := (*h)[n]
	*h = (*h)[:n]
	return x
}

// ── maxHeap (resultados) ──────────────────────────────────────────────────────

func maxHeapUp(h []cand, i int) {
	for i > 0 {
		parent := (i - 1) >> 1
		if h[parent].d >= h[i].d {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
}

func maxHeapDown(h []cand, i, n int) {
	for {
		left := (i << 1) + 1
		if left >= n {
			break
		}
		j := left
		if right := left + 1; right < n && h[right].d > h[left].d {
			j = right
		}
		if h[i].d >= h[j].d {
			break
		}
		h[i], h[j] = h[j], h[i]
		i = j
	}
}

func maxHeapPush(h *[]cand, c cand) {
	*h = append(*h, c)
	maxHeapUp(*h, len(*h)-1)
}

func maxHeapPop(h *[]cand) cand {
	n := len(*h) - 1
	(*h)[0], (*h)[n] = (*h)[n], (*h)[0]
	maxHeapDown(*h, 0, n)
	x := (*h)[n]
	*h = (*h)[:n]
	return x
}

// beamSearchL0 realiza a busca na camada 0 com beam width ef, partindo de ep.
// Recebe hb com slices pré-alocados para evitar alocações no hot path.
// Retorna até k melhores candidatos em hb.res (ordenados por distância crescente).
// ATENÇÃO: o slice retornado aponta para hb.res — inválido após a próxima chamada com o mesmo hb.
func (idx *Index) beamSearchL0(ep int, q *[dim]int8, ef, k int, hb *heapBufs) []cand {
	epD := distSq(q, idx.vecAt(ep))

	// Reutiliza slices pré-alocados; minHeap/maxHeap são inline sem boxing.
	hb.cands = append(hb.cands[:0], cand{ep, epD})
	hb.res = append(hb.res[:0], cand{ep, epD})

	stride := 2 * idx.m
	stride3 := stride * 3 // stride em bytes (3 bytes por vizinho uint24)
	vbs := <-idx.visited
	defer func() {
		for _, n := range vbs.toReset {
			vbs.bits[n>>6] &^= 1 << uint(n&63)
		}
		vbs.toReset = vbs.toReset[:0]
		idx.visited <- vbs
	}()
	vbs.bits[ep>>6] |= 1 << uint(ep&63)
	vbs.toReset = append(vbs.toReset, ep)

	for len(hb.cands) > 0 {
		c := minHeapPop(&hb.cands)

		// Termina se o melhor candidato restante é pior que o pior resultado
		if len(hb.res) >= ef && c.d > hb.res[0].d {
			break
		}

		cnt := int(idx.adj0Cnt[c.idx])
		base := c.idx * stride3
		for j := 0; j < cnt; j++ {
			p := base + j*3
			nbr := int(idx.adj0[p]) | int(idx.adj0[p+1])<<8 | int(idx.adj0[p+2])<<16
			if vbs.bits[nbr>>6]>>uint(nbr&63)&1 != 0 {
				continue
			}
			vbs.bits[nbr>>6] |= 1 << uint(nbr&63)
			vbs.toReset = append(vbs.toReset, nbr)

			d := distSq(q, idx.vecAt(nbr))
			if len(hb.res) < ef || d < hb.res[0].d {
				minHeapPush(&hb.cands, cand{nbr, d})
				maxHeapPush(&hb.res, cand{nbr, d})
				if len(hb.res) > ef {
					maxHeapPop(&hb.res)
				}
			}
		}
	}

	// Reverte o maxHeap (maior distância no topo → menor na frente) e trunca em k.
	// Sem alocações: opera diretamente sobre hb.res.
	n := len(hb.res)
	for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
		hb.res[i], hb.res[j] = hb.res[j], hb.res[i]
	}
	if n > k {
		n = k
	}
	return hb.res[:n]
}

// Search encontra os k vizinhos mais próximos de query e retorna a fração
// de vizinhos fraudulentos (fraud score em [0.0, 1.0]).
//
// Pipeline de 3 fases:
//  1. Estágio 1 (ef=efFast): busca int8 rápida.
//  2. Estágio 2 (ef=efSearch): apenas se resultado for limítrofe (2 ou 3 de k=5).
//  3. Rerank float32: recalcula distâncias dos k candidatos com a query
//     original (float32), antes da quantização int8. Corrige o erro de
//     arredondamento da quantização e melhora o recall, especialmente nos
//     casos em que vizinhos estão a distâncias muito próximas.
func (idx *Index) Search(query []float32, k int) float32 {
	q := quantizeQuery(query)
	ep := idx.entry

	// Descende pelas camadas superiores (topo → camada 1)
	for l := len(idx.upper) - 1; l >= 0; l-- {
		ep = idx.greedyDescend(&idx.upper[l], ep, &q)
	}

	// Adquire heapBufs pré-alocado; os slices internos são reutilizados
	// pelas duas chamadas a beamSearchL0 sem novas alocações de heap.
	hb := <-idx.heaps
	defer func() { idx.heaps <- hb }()

	// Estágio 1: busca rápida com efFast
	results := idx.beamSearchL0(ep, &q, efFast, k, hb)

	// Contagem inicial pós stage-1
	var fraudCount int
	for _, r := range results {
		if idx.labels[r.idx] == 1 {
			fraudCount++
		}
	}

	// Retorno antecipado para casos claramente aprovados ou rejeitados.
	// Com k=5 e threshold=0.6: limítrofes são 2/5 (0.4) e 3/5 (0.6).
	if fraudCount != 2 && fraudCount != 3 {
		// Rerank float32 mesmo em casos claros: reconstrói com precisão
		// máxima os k vizinhos (custo: k=5 chamadas a distSqFloat — trivial).
		return idx.rerankFloat(query, results, k)
	}

	// Estágio 2: refina resultado limítrofe com ef maior.
	results = idx.beamSearchL0(ep, &q, efSearch, k, hb)
	return idx.rerankFloat(query, results, k)
}

// rerankFloat recalcula as distâncias dos candidatos usando a query float32
// original (sem quantização) e devolve a fração de fraudes entre os k mais próximos.
// Custo: k=5 chamadas a distSqFloat + insertion sort O(k²=25) — negligível vs. beam search.
func (idx *Index) rerankFloat(query []float32, results []cand, k int) float32 {
	n := len(results)
	if n == 0 {
		return 0
	}
	if n > k {
		n = k
	}

	// Recalcula distâncias em float32 (query original, sem erro de quantização).
	// ranked é stack-allocated: k=5 entradas, zero heap.
	type rankCand struct {
		nodeIdx int
		d       float32
	}
	var ranked [5]rankCand
	for i := 0; i < n; i++ {
		ranked[i] = rankCand{results[i].idx, distSqFloat(query, idx.vecAt(results[i].idx))}
	}

	// Insertion sort para n=5: O(k²)=O(25) — mais rápido que qualquer sort.Slice.
	for i := 1; i < n; i++ {
		x := ranked[i]
		j := i - 1
		for j >= 0 && ranked[j].d > x.d {
			ranked[j+1] = ranked[j]
			j--
		}
		ranked[j+1] = x
	}

	fraudCount := 0
	for i := 0; i < n; i++ {
		if idx.labels[ranked[i].nodeIdx] == 1 {
			fraudCount++
		}
	}
	return float32(fraudCount) / float32(k)
}
