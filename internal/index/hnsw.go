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
	"math"
	"os"
	"time"
)

const (
	dim      = 14
	efSearch = 20 // candidatos na busca do nível 0
)

// Index é o índice HNSW carregado em memória.
type Index struct {
	n, m, nLayers, entry int

	vectors []int8   // flat, stride dim: nó i → vectors[i*dim:(i+1)*dim]
	labels  []uint8  // labels[i] = 0 (legit) ou 1 (fraud)
	adj0    []uint32 // flat, stride 2*M: adj0[i*stride:i*stride+adj0Cnt[i]]
	adj0Cnt []uint8  // adj0Cnt[i] = vizinhos válidos do nó i na camada 0
	upper   []upperLayer // upper[l] = camada l+1 do grafo HNSW
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

	br := bufio.NewReaderSize(f, 8<<20)

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

	// Vetores quantizados
	vectors := make([]int8, N*dim)
	for i := range vectors {
		b, _ := br.ReadByte()
		vectors[i] = int8(b)
	}

	// Labels
	labels := make([]uint8, N)
	io.ReadFull(br, labels) //nolint:errcheck

	// Camada 0: stride fixo
	adj0 := make([]uint32, N*stride)
	for i := range adj0 {
		adj0[i] = r32()
	}
	adj0Cnt := make([]uint8, N)
	io.ReadFull(br, adj0Cnt) //nolint:errcheck

	// Camadas superiores (l=1..nLayers-1): CSR
	upper := make([]upperLayer, 0, nLayers-1)
	for l := 1; l < nLayers; l++ {
		Nl := int(r32())
		nodes := make([]int32, Nl)
		for i := range nodes {
			nodes[i] = int32(r32())
		}
		off := make([]uint32, Nl+1)
		for i := range off {
			off[i] = r32()
		}
		nbr := make([]uint32, off[Nl])
		for i := range nbr {
			nbr[i] = r32()
		}
		upper = append(upper, upperLayer{nodes: nodes, off: off, nbr: nbr})
	}

	log.Printf("índice HNSW carregado: %d nós, M=%d, %d camadas em %s",
		N, M, nLayers, time.Since(start))

	return &Index{
		n: N, m: M, nLayers: nLayers, entry: entry,
		vectors: vectors, labels: labels,
		adj0: adj0, adj0Cnt: adj0Cnt,
		upper: upper,
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
			v[i] = int8(math.Round(float64(r)))
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

// beamSearchL0 realiza a busca na camada 0 com beam width ef,
// partindo de ep. Retorna até k melhores candidatos ordenados por distância.
func (idx *Index) beamSearchL0(ep int, q *[dim]int8, ef, k int) []cand {
	epD := distSq(q, idx.vecAt(ep))

	cands := &minCandHeap{cand{ep, epD}}
	heap.Init(cands)

	res := &maxCandHeap{cand{ep, epD}}
	heap.Init(res)

	stride := 2 * idx.m
	visited := make(map[int]struct{}, ef*4)
	visited[ep] = struct{}{}

	for cands.Len() > 0 {
		c := heap.Pop(cands).(cand)

		// Termina se o melhor candidato restante é pior que o pior resultado
		if res.Len() >= ef && c.d > (*res)[0].d {
			break
		}

		cnt := int(idx.adj0Cnt[c.idx])
		base := c.idx * stride
		for j := 0; j < cnt; j++ {
			nbr := int(idx.adj0[base+j])
			if _, seen := visited[nbr]; seen {
				continue
			}
			visited[nbr] = struct{}{}

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

	// Extrai os k melhores em ordem crescente de distância.
	// O maxCandHeap tem o pior no topo; invertemos para ordenar crescente.
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
func (idx *Index) Search(query []float32, k int) float32 {
	q := quantizeQuery(query)
	ep := idx.entry

	// Descende pelas camadas superiores (topo → camada 1)
	for l := len(idx.upper) - 1; l >= 0; l-- {
		ep = idx.greedyDescend(&idx.upper[l], ep, &q)
	}

	// Beam search na camada 0
	results := idx.beamSearchL0(ep, &q, efSearch, k)

	var fraudCount float32
	for _, r := range results {
		if idx.labels[r.idx] == 1 {
			fraudCount++
		}
	}
	return fraudCount / float32(k)
}
