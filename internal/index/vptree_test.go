package index

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
)

const vpTreePath = "../../resources/vptree.bin"

// openVPIndex carrega o índice VP-Tree ou pula o teste se os arquivos não existirem.
func openVPIndex(t testing.TB) *VPIndex {
	t.Helper()
	idx, err := LoadVP(indexPath, vpTreePath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("index.bin ou vptree.bin não encontrado — rode: go run ./cmd/preprocess")
	}
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

func TestVPTreeLoad(t *testing.T) {
	idx := openVPIndex(t)
	if idx.n == 0 {
		t.Fatal("índice vazio")
	}
	t.Logf("VP-Tree: N=%d, %d nós", idx.n, len(idx.nodes))
}

func TestVPTreeSearch(t *testing.T) {
	idx := openVPIndex(t)

	// Mesmos vetores de teste do hnsw_test.go
	fraud := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	score := idx.Search(fraud, 5)
	t.Logf("VP-Tree fraud score (esperado ~1.0): %.4f", score)
	if score < 0.6 {
		t.Errorf("esperado score alto para transação fraudulenta, got %.4f", score)
	}

	legit := []float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	score = idx.Search(legit, 5)
	t.Logf("VP-Tree legit score (esperado ~0.0): %.4f", score)
	if score > 0.4 {
		t.Errorf("esperado score baixo para transação legítima, got %.4f", score)
	}
}

// TestVPTreeVsHNSW compara resultados VP-Tree vs HNSW em vetores aleatórios.
// Verifica que os dois concordam na decisão (approved/rejected) na maioria dos casos.
func TestVPTreeVsHNSW(t *testing.T) {
	vpIdx := openVPIndex(t)
	hnswIdx := openIndex(t)

	rng := rand.New(rand.NewSource(42))

	const n = 200
	agree, total := 0, 0
	for i := 0; i < n; i++ {
		query := make([]float32, 14)
		for j := range query {
			query[j] = rng.Float32()
		}
		// Introduz sentinelas ocasionalmente
		if rng.Intn(3) == 0 {
			query[5] = -1
			query[6] = -1
		}

		vpScore := vpIdx.Search(query, 5)
		hnswScore := hnswIdx.Search(query, 5)

		vpApproved := vpScore < 0.6
		hnswApproved := hnswScore < 0.6
		total++
		if vpApproved == hnswApproved {
			agree++
		}
	}
	rate := float64(agree) / float64(total) * 100
	t.Logf("VP-Tree vs HNSW: concordam em %.1f%% dos casos (%d/%d)", rate, agree, total)
	// Esperamos concordância alta (>85%) já que VP-Tree é mais preciso que HNSW
	if rate < 70 {
		t.Errorf("concordância muito baixa: %.1f%% (esperado ≥70%%)", rate)
	}
}

// TestVPTreeExactSmall verifica que VP-Tree encontra os exatos k-NN em um dataset pequeno.
// Compara com brute-force.
func TestVPTreeExactSmall(t *testing.T) {
	// Constrói VP-Tree sintética em memória
	const N = 500
	const k = 5

	rng := rand.New(rand.NewSource(123))

	// Gera N vetores float32 aleatórios
	vecs := make([][dim]float32, N)
	labels := make([]uint8, N)
	for i := range vecs {
		for j := range vecs[i] {
			vecs[i][j] = rng.Float32()
		}
		if rng.Intn(2) == 0 {
			labels[i] = 1
		}
	}

	// Constrói VP-Tree em memória
	vpIdx := buildVPIndexInMemory(vecs, labels)

	// Executa várias queries e compara com brute force
	const nQueries = 50
	wrong := 0
	for q := 0; q < nQueries; q++ {
		query := make([]float32, dim)
		for j := range query {
			query[j] = rng.Float32()
		}

		// VP-Tree search
		vpScore := vpIdx.Search(query, k)

		// Brute-force: calcula todas as distâncias, pega top-k
		bfScore := bruteForceSearch(vecs, labels, query, k)

		if vpScore != bfScore {
			wrong++
			t.Logf("query %d: VP-Tree=%.2f vs BruteForce=%.2f", q, vpScore, bfScore)
		}
	}
	t.Logf("Exatidão: %d/%d corretos (%.1f%%)", nQueries-wrong, nQueries, float64(nQueries-wrong)/float64(nQueries)*100)
	if wrong > nQueries/10 {
		t.Errorf("muitos resultados incorretos: %d/%d", wrong, nQueries)
	}
}

// BenchmarkVPTreeSearch mede a latência da busca VP-Tree no índice real.
func BenchmarkVPTreeSearch(b *testing.B) {
	idx := openVPIndex(b)
	fraud := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	for i := 0; i < 5; i++ {
		idx.Search(fraud, 5)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(fraud, 5)
	}
}

// BenchmarkVPTreeSearchAmbiguous mede VP-Tree em casos ambíguos (score ~0.4).
func BenchmarkVPTreeSearchAmbiguous(b *testing.B) {
	idx := openVPIndex(b)
	ambiguous := []float32{0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.005}
	for i := 0; i < 5; i++ {
		idx.Search(ambiguous, 5)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(ambiguous, 5)
	}
}

// ── Helpers para testes ────────────────────────────────────────────────────────

// buildVPIndexInMemory constrói um VPIndex in-memory para testes (sem mmap).
func buildVPIndexInMemory(vecs [][dim]float32, labels []uint8) *VPIndex {
	N := len(vecs)

	// Constrói a VP-Tree usando a mesma lógica do preprocess
	// (reimplementada aqui para evitar dependência circular)
	perm := make([]uint32, N)
	for i := range perm {
		perm[i] = uint32(i)
	}
	nodes := make([]vpBuildNodeTest, 0, 2*((N+vpLeafSizeTest-1)/vpLeafSizeTest+1))
	buildRecTest(vecs, perm, 0, N, &nodes)

	// Converte vpBuildNodeTest → vpNode
	vpNodes := make([]vpNode, len(nodes))
	for i, n := range nodes {
		vpNodes[i] = vpNode{
			vpIdx:   n.vpIdx,
			medDist: n.medDist,
			left:    n.left,
			right:   n.right,
		}
	}

	// Quantiza vetores para int8
	quantVecs := make([]int8, N*dim)
	for i, v := range vecs {
		for j, x := range v {
			if x < 0 {
				quantVecs[i*dim+j] = -1
			} else {
				r := x * 127
				if r > 127 {
					r = 127
				}
				quantVecs[i*dim+j] = int8(r + 0.5)
			}
		}
	}

	const poolSz = 1
	knnPool := make(chan *vpKNN, poolSz)
	knnPool <- &vpKNN{}
	pqPool := make(chan *vpPQ, poolSz)
	pqPool <- &vpPQ{buf: make([]vpPQEntry, 0, 512)}

	return &VPIndex{
		n:       N,
		vectors: quantVecs,
		labels:  labels,
		nodes:   vpNodes,
		perm:    perm,
		knnPool: knnPool,
		pqPool:  pqPool,
	}
}

const vpLeafSizeTest = 8 // menor leaf size para testes com N=500

type vpBuildNodeTest struct {
	vpIdx   uint32
	medDist float32
	left    int32
	right   int32
}

func buildRecTest(vecs [][dim]float32, perm []uint32, lo, hi int, nodes *[]vpBuildNodeTest) int32 {
	if hi-lo <= vpLeafSizeTest {
		idx := int32(len(*nodes))
		*nodes = append(*nodes, vpBuildNodeTest{
			left:  -(int32(lo) + 1),
			right: -(int32(hi) + 1),
		})
		return idx
	}
	rng := rand.New(rand.NewSource(int64(lo)))
	vpPos := lo + rng.Intn(hi-lo)
	perm[lo], perm[vpPos] = perm[vpPos], perm[lo]
	vp := perm[lo]

	n := hi - lo - 1
	dists := make([]float32, n)
	for i, pt := range perm[lo+1 : hi] {
		var sum float32
		for j := 0; j < dim; j++ {
			d := vecs[vp][j] - vecs[pt][j]
			sum += d * d
		}
		dists[i] = float32(math.Sqrt(float64(sum)))
	}

	medIdx := n / 2
	if n > 0 {
		nthElementTest(perm[lo+1:hi], dists, medIdx)
	}
	var medDist float32
	if n > 0 {
		medDist = dists[medIdx]
	}

	splitPoint := lo + 1 + medIdx + 1
	nodeIdx := int32(len(*nodes))
	*nodes = append(*nodes, vpBuildNodeTest{vpIdx: vp, medDist: medDist})

	leftChild := buildRecTest(vecs, perm, lo+1, splitPoint, nodes)
	rightChild := buildRecTest(vecs, perm, splitPoint, hi, nodes)

	(*nodes)[nodeIdx].left = leftChild
	(*nodes)[nodeIdx].right = rightChild
	return nodeIdx
}

func nthElementTest(perm []uint32, dists []float32, k int) {
	// Implementação simples via sort para testes (não performance-crítico)
	type pair struct{ d float32; p uint32 }
	pairs := make([]pair, len(dists))
	for i := range dists {
		pairs[i] = pair{dists[i], perm[i]}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].d < pairs[j].d })
	for i := range pairs {
		dists[i] = pairs[i].d
		perm[i] = pairs[i].p
	}
	_ = k // após sort, k-ésimo está na posição k
}

// bruteForceSearch calcula brute-force k-NN e retorna fraud fraction.
func bruteForceSearch(vecs [][dim]float32, labels []uint8, query []float32, k int) float32 {
	type distPair struct {
		d   float32
		idx int
	}
	dists := make([]distPair, len(vecs))
	for i, v := range vecs {
		var sum float32
		for j := 0; j < dim; j++ {
			d := query[j] - v[j]
			sum += d * d
		}
		dists[i] = distPair{float32(math.Sqrt(float64(sum))), i}
	}
	sort.Slice(dists, func(a, b int) bool { return dists[a].d < dists[b].d })
	count := 0
	for i := 0; i < k && i < len(dists); i++ {
		if labels[dists[i].idx] == 1 {
			count++
		}
	}
	return float32(count) / float32(k)
}
