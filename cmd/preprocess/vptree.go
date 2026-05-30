// VP-Tree builder para o preprocess.
//
// Constrói uma VP-Tree (Vantage Point Tree) a partir dos vetores float32 brutos
// e serializa em vptree.bin para uso pelo servidor de API.
//
// Formato de saída (vptree.bin):
//
//	[4] uint32 N          — total de pontos
//	[4] uint32 nodeCount  — total de nós na árvore
//	[4] uint32 leafSize   — threshold de folha usado na construção
//	[4] uint32 padding    — alinhamento
//	[nodeCount × 16] vpNode — nós da árvore (internos e folhas)
//	[N × 4] uint32          — perm[]: índices originais na ordem VP-Tree
//	[N × 14 × 2] int16      — vetores quantizados em int16 (alta precisão)
//
// Codificação de folha em vpNode:
//
//	left  = -(lo+1)  → lo = -(left+1)
//	right = -(hi+1)  → hi = -(right+1)
//	range perm[lo:hi] contém os pontos desta folha.
//
// Quantização int16:
//
//	Valores em [0,1] → [0, 32767]. Sentinela −1.0 → int16(−1).
//	Precisão: ±0.000015 por dimensão (vs ±0.004 do int8).
//	Erro máximo L2 em 14D: sqrt(14 × (0.000015/2)²) ≈ 0.000028.
//	Elimina virtualmente todos os erros de classificação por quantização.
package main

import (
	"bufio"
	"encoding/binary"
	"log"
	"math"
	"math/rand"
	"os"
	"time"
)

const vpLeafSize = 64 // pontos por folha — cache-friendly para L1/L2

// vpBuildNode tem o mesmo layout binário que internal/index.vpNode (16 bytes).
type vpBuildNode struct {
	vpIdx   uint32  // índice do vantage point
	medDist float32 // distância mediana para partição
	left    int32   // filho esquerdo (≥0) ou -(lo+1) se folha
	right   int32   // filho direito  (≥0) ou -(hi+1) se folha
}

// distEuclid calcula a distância euclidiana real entre dois vetores float32.
func distEuclid(a, b [dim]float32) float32 {
	var sum float32
	for i := 0; i < dim; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

// nthElement rearraneja perm[0:n] e dists[0:n] (em paralelo) de forma que
// dists[k] seja o k-ésimo menor elemento (0-indexado).
// Todos os elementos antes de k são ≤ dists[k]; todos depois são ≥ dists[k].
// Complexidade: O(n) amortizado com pivot mediana-de-três.
func nthElement(perm []uint32, dists []float32, k int) {
	lo, hi := 0, len(dists)-1
	for lo < hi {
		// Pivot: mediana-de-três para evitar pior caso em dados ordenados
		mid := (lo + hi) / 2
		if dists[mid] < dists[lo] {
			perm[lo], perm[mid] = perm[mid], perm[lo]
			dists[lo], dists[mid] = dists[mid], dists[lo]
		}
		if dists[hi] < dists[lo] {
			perm[lo], perm[hi] = perm[hi], perm[lo]
			dists[lo], dists[hi] = dists[hi], dists[lo]
		}
		if dists[mid] < dists[hi] {
			perm[hi], perm[mid] = perm[mid], perm[hi]
			dists[hi], dists[mid] = dists[mid], dists[hi]
		}
		pivot := dists[hi]

		// Partição de Lomuto: move elementos < pivot para a esquerda
		i := lo
		for j := lo; j < hi; j++ {
			if dists[j] < pivot {
				perm[i], perm[j] = perm[j], perm[i]
				dists[i], dists[j] = dists[j], dists[i]
				i++
			}
		}
		perm[i], perm[hi] = perm[hi], perm[i]
		dists[i], dists[hi] = dists[hi], dists[i]

		if i == k {
			return // pivot está na posição certa
		} else if i < k {
			lo = i + 1 // k está na metade direita
		} else {
			hi = i - 1 // k está na metade esquerda
		}
	}
}

// buildVPTree constrói a VP-Tree a partir dos vetores float32 brutos.
// seed controla a seleção aleatória de vantage points — seeds diferentes produzem
// árvores com estruturas diferentes, útil para busca dual-tree.
//
// Complexidade de build: O(N log N) — quickselect O(N) por nível × O(log N) níveis.
// Memória de pico: ~2N floats = ~24 MB para N=3M (alocações por nível, liberadas pelo GC).
func buildVPTree(vecs [][dim]float32, seed int64) (nodes []vpBuildNode, perm []uint32) {
	N := len(vecs)
	perm = make([]uint32, N)
	for i := range perm {
		perm[i] = uint32(i)
	}
	// Estimativa de nós: ~2 × ceil(N/vpLeafSize) para árvore binária balanceada
	nodes = make([]vpBuildNode, 0, 2*((N+vpLeafSize-1)/vpLeafSize+1))

	rng := rand.New(rand.NewSource(seed))
	buildRec(vecs, perm, 0, N, &nodes, rng)
	return nodes, perm
}

// buildRec constrói recursivamente a VP-Tree para o range perm[lo:hi].
// Retorna o índice do nó raiz desta subárvore em *nodes.
//
// Invariante de folha:
//   - hi-lo ≤ vpLeafSize → nó folha com range perm[lo:hi]
//
// Invariante de nó interno:
//   - VP = perm[lo] (selecionado aleatoriamente e movido para lo)
//   - perm[lo+1 : splitPoint] = pontos com dist(vp,x) ≤ medDist (left)
//   - perm[splitPoint : hi]   = pontos com dist(vp,x) ≥ medDist (right)
func buildRec(vecs [][dim]float32, perm []uint32, lo, hi int, nodes *[]vpBuildNode, rng *rand.Rand) int32 {
	if hi-lo <= vpLeafSize {
		// Folha: codifica range como valores negativos
		idx := int32(len(*nodes))
		*nodes = append(*nodes, vpBuildNode{
			left:  -(int32(lo) + 1),
			right: -(int32(hi) + 1),
		})
		return idx
	}

	// Seleciona vantage point aleatório em [lo, hi) e move para perm[lo]
	vpPos := lo + rng.Intn(hi-lo)
	perm[lo], perm[vpPos] = perm[vpPos], perm[lo]
	vp := perm[lo]

	// Calcula distâncias do vp para todos os outros pontos em [lo+1, hi)
	n := hi - lo - 1
	dists := make([]float32, n)
	for i, pt := range perm[lo+1 : hi] {
		dists[i] = distEuclid(vecs[vp], vecs[pt])
	}

	// Encontra mediana via quickselect e particiona
	medIdx := n / 2
	if n > 0 {
		nthElement(perm[lo+1:hi], dists, medIdx)
	}
	var medDist float32
	if n > 0 {
		medDist = dists[medIdx]
	}

	// splitPoint: primeiro índice do filho direito
	// [lo+1, splitPoint) = filho esquerdo (medIdx+1 pontos, dist ≤ medDist)
	// [splitPoint, hi)   = filho direito  (n-medIdx-1 pontos, dist ≥ medDist)
	splitPoint := lo + 1 + medIdx + 1

	// Reserva slot para este nó (filhos serão preenchidos após recursão)
	nodeIdx := int32(len(*nodes))
	*nodes = append(*nodes, vpBuildNode{vpIdx: vp, medDist: medDist})

	leftChild := buildRec(vecs, perm, lo+1, splitPoint, nodes, rng)
	rightChild := buildRec(vecs, perm, splitPoint, hi, nodes, rng)

	// Preenche filhos (slice pode ter crescido, mas índice é estável)
	(*nodes)[nodeIdx].left = leftChild
	(*nodes)[nodeIdx].right = rightChild

	return nodeIdx
}

// quantize16 converte float32 → int16.
// Valores em [0,1] → [0, 32767]. Sentinela −1.0 → int16(−1).
// Precisão: ±0.000015 por dimensão (250× melhor que int8).
func quantize16(v float32) int16 {
	if v < -0.5 { // único caso negativo: sentinela −1.0
		return -1
	}
	if v < 0 {
		return 0
	}
	r := math.Round(float64(v) * 32767)
	if r > 32767 {
		r = 32767
	}
	return int16(r)
}

// writeVPTree serializa as duas VP-Trees em vptree.bin (formato dual-tree).
//
// Formato do arquivo:
//
//	[4] uint32 N             — total de pontos
//	[4] uint32 nodeCount1    — nós da árvore 1
//	[4] uint32 leafSize      — threshold de folha
//	[4] uint32 nodeCount2    — nós da árvore 2 (era "padding")
//	[nodeCount1 × 16] vpNode — nós da árvore 1
//	[N × 4] uint32           — perm1
//	[nodeCount2 × 16] vpNode — nós da árvore 2
//	[N × 4] uint32           — perm2
//	[N × dim × 2] int16      — vetores em int16 (alta precisão)
//
// Inclui os vetores float32 originais quantizados em int16 (alta precisão)
// para garantir que as distâncias computadas na busca sejam equivalentes
// às distâncias float32 brutas (dentro de ±0.000028 em L2 para 14D).
func writeVPTree(path string, N int, nodes1 []vpBuildNode, perm1 []uint32, nodes2 []vpBuildNode, perm2 []uint32, vecs [][dim]float32) error {
	start := time.Now()
	log.Printf("escrevendo VP-Tree dual em %s (%d+%d nós, %d pontos)...", path, len(nodes1), len(nodes2), N)

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	bw := bufio.NewWriterSize(f, 8<<20) // 8 MB buffer

	var buf [4]byte

	write32 := func(v uint32) {
		binary.LittleEndian.PutUint32(buf[:], v)
		bw.Write(buf[:]) //nolint:errcheck
	}
	writeF32 := func(v float32) {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		bw.Write(buf[:]) //nolint:errcheck
	}
	writeI32 := func(v int32) {
		binary.LittleEndian.PutUint32(buf[:], uint32(v))
		bw.Write(buf[:]) //nolint:errcheck
	}
	writeI16 := func(v int16) {
		bw.WriteByte(byte(v))
		bw.WriteByte(byte(v >> 8))
	}
	writeNodes := func(nodes []vpBuildNode) {
		for _, nd := range nodes {
			write32(nd.vpIdx)
			writeF32(nd.medDist)
			writeI32(nd.left)
			writeI32(nd.right)
		}
	}
	writePerm := func(perm []uint32) {
		for _, p := range perm {
			write32(p)
		}
	}

	// Cabeçalho (16 bytes)
	write32(uint32(N))
	write32(uint32(len(nodes1)))
	write32(vpLeafSize)
	write32(uint32(len(nodes2))) // era "padding=0"

	// Árvore 1: nós + permutação
	writeNodes(nodes1)
	writePerm(perm1)

	// Árvore 2: nós + permutação
	writeNodes(nodes2)
	writePerm(perm2)

	// Vetores int16 (N × dim × 2 bytes) — alta precisão para busca exata
	for _, v := range vecs {
		for _, x := range v {
			writeI16(quantize16(x))
		}
	}

	if err := bw.Flush(); err != nil {
		return err
	}

	stat, _ := os.Stat(path)
	log.Printf("VP-Tree dual salva: %s (%.1f MB) em %s",
		path, float64(stat.Size())/1024/1024, time.Since(start))
	return nil
}
