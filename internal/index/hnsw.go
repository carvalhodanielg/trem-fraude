// Package index implementa busca HNSW sobre o índice compacto gerado
// pelo cmd/preprocess. O índice usa vetores quantizados em int8 e
// adjacência em arrays flat (sem maps), cabendo em ~138 MB para 3M nós.
package index

import (
	"bufio"
	"container/heap"
	"encoding/binary"
	"io"
	"log"
	"os"
	"time"
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

	vectors []int8       // flat, stride dim: nó i → vectors[i*dim:(i+1)*dim]
	labels  []uint8      // labels[i] = 0 (legit) ou 1 (fraud)
	adj0    []byte       // flat, 3 bytes/vizinho (uint24 LE): nó i → adj0[i*stride*3:(i*stride+adj0Cnt[i])*3]
	adj0Cnt []uint8      // adj0Cnt[i] = vizinhos válidos do nó i na camada 0
	upper   []upperLayer // upper[l] = camada l+1 do grafo HNSW

	// visited é um canal de bitsets pré-alocados usados no beam search.
	// Canal em vez de sync.Pool: o canal mantém referências vivas aos bitsets,
	// evitando que o GC os evicte e force realocações de 375 KB por requisição.
	// Com GOMAXPROCS=1, ao mais poucos goroutines seguram um bitset simultaneamente;
	// o canal bloqueia se todos estiverem em uso (backpressure sem deadlock).
	visited chan *visitedBitset

	// heaps é um canal de heapBufs pré-alocados, mesmo padrão que visited.
	// Elimina alocações de minCandHeap e maxCandHeap a cada requisição.
	heaps chan *heapBufs
}

// upperLayer armazena a adjacência de uma camada superior (l >= 1) em CSR.
type upperLayer struct {
	nodes []int32  // nodeIdxs ordenados crescente (Nl entradas)
	off   []uint32 // offsets CSR (Nl+1 entradas)
	nbr   []uint32 // vizinhos (off[Nl] entradas)
}

// Load lê o binário compacto gerado pelo cmd/preprocess.
func Load(path string) (*Index, error) {
	start := time.Now()
	log.Println("carregando índice HNSW...")

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Buffer menor: 512KB economiza ~7.5 MB vs 8 MB durante o loading.
	br := bufio.NewReaderSize(f, 512<<10)

	r32 := func() uint32 {
		var buf [4]byte
		io.ReadFull(br, buf[:]) //nolint:errcheck
		return binary.LittleEndian.Uint32(buf[:])
	}

	N := int(r32())
	M := int(r32())
	nLayers := int(r32())
	entry := int(r32())
	stride := 2 * M

	// Vetores quantizados — binary.Read lê int8 byte-a-byte sem alocações extras.
	vectors := make([]int8, N*dim)
	if err := binary.Read(br, binary.LittleEndian, vectors); err != nil {
		return nil, err
	}

	// Labels
	labels := make([]uint8, N)
	io.ReadFull(br, labels) //nolint:errcheck

	// Camada 0: stride fixo — 3 bytes por vizinho (uint24 LE), N < 2^24
	adj0 := make([]byte, N*stride*3)
	if _, err := io.ReadFull(br, adj0); err != nil {
		return nil, err
	}
	adj0Cnt := make([]uint8, N)
	io.ReadFull(br, adj0Cnt) //nolint:errcheck

	// Camadas superiores (l=1..nLayers-1): CSR
	upper := make([]upperLayer, 0, nLayers-1)
	for l := 1; l < nLayers; l++ {
		Nl := int(r32())
		nodes := make([]int32, Nl)
		if err := binary.Read(br, binary.LittleEndian, nodes); err != nil {
			return nil, err
		}
		off := make([]uint32, Nl+1)
		if err := binary.Read(br, binary.LittleEndian, off); err != nil {
			return nil, err
		}
		nbr := make([]uint32, off[Nl])
		if err := binary.Read(br, binary.LittleEndian, nbr); err != nil {
			return nil, err
		}
		upper = append(upper, upperLayer{nodes: nodes, off: off, nbr: nbr})
	}

	log.Printf("índice HNSW carregado: %d nós, M=%d, %d camadas em %s",
		N, M, nLayers, time.Since(start))

	// Pré-aloca 32 bitsets no canal. Cada bitset ocupa (N+63)/64*8 ≈ 375 KB
	// para N=3M. Total: 32×375KB ≈ 11.7 MB.
	// Com GOMAXPROCS=1, apenas 1 bitset é usado simultaneamente; 32 é margem ampla.
	bitWords := (N + 63) / 64
	const poolSize = 32
	visitedCh := make(chan *visitedBitset, poolSize)
	for i := 0; i < poolSize; i++ {
		visitedCh <- &visitedBitset{
			bits:    make([]uint64, bitWords),
			toReset: make([]int, 0, 512),
		}
	}

	// Pré-aloca heapBufs para eliminar alocações no hot path.
	// cands pode crescer até stride*efSearch = 6*60 = 360 elementos (pior caso M=3).
	// res é limitado a efSearch = 60 elementos.
	// Com GOMAXPROCS=1, apenas 1 heapBufs é usado simultaneamente; pool=32 é seguro.
	candsCap := stride*efSearch + 16
	const heapPoolSize = 32
	heapsCh := make(chan *heapBufs, heapPoolSize)
	for i := 0; i < heapPoolSize; i++ {
		heapsCh <- &heapBufs{
			cands: make([]cand, 0, candsCap),
			res:   make([]cand, 0, efSearch+1),
		}
	}

	return &Index{
		n: N, m: M, nLayers: nLayers, entry: entry,
		vectors: vectors, labels: labels,
		adj0: adj0, adj0Cnt: adj0Cnt,
		upper:   upper,
		visited: visitedCh,
		heaps:   heapsCh,
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

// ── Heaps para o beam search ──────────────────────────────────────────────

type cand struct {
	idx int
	d   int32
}

// minCandHeap: menor distância no topo (candidatos a explorar).
type minCandHeap []cand

func (h minCandHeap) Len() int            { return len(h) }
func (h minCandHeap) Less(i, j int) bool { return h[i].d < h[j].d }
func (h minCandHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minCandHeap) Push(x any)        { *h = append(*h, x.(cand)) }
func (h *minCandHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// maxCandHeap: maior distância no topo (descarta os piores do resultado).
type maxCandHeap []cand

func (h maxCandHeap) Len() int            { return len(h) }
func (h maxCandHeap) Less(i, j int) bool { return h[i].d > h[j].d }
func (h maxCandHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxCandHeap) Push(x any)        { *h = append(*h, x.(cand)) }
func (h *maxCandHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// beamSearchL0 realiza a busca na camada 0 com beam width ef, partindo de ep.
// Recebe hb com slices pré-alocados para evitar alocações no hot path.
// Retorna até k melhores candidatos em hb.res (ordenados por distância crescente).
// ATENÇÃO: o slice retornado aponta para hb.res — inválido após a próxima chamada com o mesmo hb.
func (idx *Index) beamSearchL0(ep int, q *[dim]int8, ef, k int, hb *heapBufs) []cand {
	epD := distSq(q, idx.vecAt(ep))

	// Reutiliza slices pré-alocados sem novas alocações.
	hb.cands = append(hb.cands[:0], cand{ep, epD})
	cands := (*minCandHeap)(&hb.cands)
	heap.Init(cands)

	hb.res = append(hb.res[:0], cand{ep, epD})
	res := (*maxCandHeap)(&hb.res)
	heap.Init(res)

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

	for cands.Len() > 0 {
		c := heap.Pop(cands).(cand)

		// Termina se o melhor candidato restante é pior que o pior resultado
		if res.Len() >= ef && c.d > (*res)[0].d {
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
			if res.Len() < ef || d < (*res)[0].d {
				heap.Push(cands, cand{nbr, d})
				heap.Push(res, cand{nbr, d})
				if res.Len() > ef {
					heap.Pop(res)
				}
			}
		}
	}

	// Reverte o maxHeap (maior distância no topo → menor na frente) e trunca em k.
	// Sem alocações: opera diretamente sobre hb.res.
	slice := []cand(*res)
	for i, j := 0, len(slice)-1; i < j; i, j = i+1, j-1 {
		slice[i], slice[j] = slice[j], slice[i]
	}
	if len(slice) > k {
		slice = slice[:k]
	}
	return slice
}

// Search encontra os k vizinhos mais próximos de query e retorna a fração
// de vizinhos fraudulentos (fraud score em [0.0, 1.0]).
//
// Usa busca adaptativa em dois estágios com k=5 e threshold=0.6:
// - Estágio 1 (ef=efFast): rápido para casos claros (score ∈ {0.0, 0.2, 0.8, 1.0})
// - Estágio 2 (ef=efSearch): refino apenas para casos limítrofes (score ∈ {0.4, 0.6})
// Os valores 2 e 3 abaixo são específicos para k=5; ajustar se k mudar.
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
	var fraudCount int
	for _, r := range results {
		if idx.labels[r.idx] == 1 {
			fraudCount++
		}
	}

	// Retorno antecipado para casos claramente aprovados ou rejeitados.
	// Com k=5 e threshold=0.6: limítrofes são 2/5 (0.4) e 3/5 (0.6).
	// Refinar apenas esses dois casos minimiza o custo do stage 2.
	if fraudCount != 2 && fraudCount != 3 {
		return float32(fraudCount) / float32(k)
	}

	// Estágio 2: refina resultado limítrofe com ef maior.
	// hb.cands e hb.res são resetados no início da chamada — sem alocações.
	results = idx.beamSearchL0(ep, &q, efSearch, k, hb)
	fraudCount = 0
	for _, r := range results {
		if idx.labels[r.idx] == 1 {
			fraudCount++
		}
	}
	return float32(fraudCount) / float32(k)
}
