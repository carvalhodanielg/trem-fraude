package index

import (
	"errors"
	"os"
	"testing"
	"time"
)

const indexPath = "../../resources/index.bin"

// openIndex carrega o índice ou pula o teste se o arquivo não existir.
// Para gerar o arquivo: go run ./cmd/preprocess
func openIndex(t testing.TB) *Index {
	t.Helper()
	idx, err := Load(indexPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("resources/index.bin não encontrado — rode: go run ./cmd/preprocess")
	}
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

func TestLoad(t *testing.T) {
	idx := openIndex(t)
	if idx == nil {
		t.Fatal("índice nulo")
	}
	if idx.n == 0 {
		t.Fatal("índice vazio")
	}
	t.Logf("nós: %d, M: %d, camadas: %d", idx.n, idx.m, idx.nLayers)
}

func TestSearch(t *testing.T) {
	idx := openIndex(t)

	// Transação fraudulenta do documento (fraud_score esperado = 1.0)
	// Vetor original float32, convertido para int8 internamente pelo Search
	fraud := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	score := idx.Search(fraud, 5)
	t.Logf("fraud score (esperado ~1.0): %.2f", score)
	if score < 0.6 {
		t.Errorf("esperado score alto para transação fraudulenta, got %.2f", score)
	}

	// Transação legítima do documento (fraud_score esperado = 0.0)
	legit := []float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	score = idx.Search(legit, 5)
	t.Logf("legit score (esperado ~0.0): %.2f", score)
	if score > 0.4 {
		t.Errorf("esperado score baixo para transação legítima, got %.2f", score)
	}
}

func TestSearchLatency(t *testing.T) {
	idx := openIndex(t)

	query := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}

	// Aquecimento (warm-up do cache da CPU)
	for i := 0; i < 5; i++ {
		idx.Search(query, 5)
	}

	iterations := 100
	start := time.Now()
	for i := 0; i < iterations; i++ {
		idx.Search(query, 5)
	}
	elapsed := time.Since(start)

	avg := elapsed / time.Duration(iterations)
	t.Logf("média por busca: %s", avg)
	t.Logf("total %d buscas: %s", iterations, elapsed)

	if avg > 50*time.Millisecond {
		t.Errorf("latência muito alta: %s (esperado < 50ms com HNSW)", avg)
	}
}

// BenchmarkSearchClear mede o hot path de casos absolutamente claros
// (fraudCount = 0 ou k=5), onde apenas o estágio 1 é executado.
func BenchmarkSearchClear(b *testing.B) {
	idx := openIndex(b)
	// Transação claramente fraudulenta (fraudCount=5, apenas stage-1)
	fraud := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	// Aquecimento
	for i := 0; i < 5; i++ {
		idx.Search(fraud, 5)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(fraud, 5)
	}
}

// BenchmarkSearchAmbiguous mede casos que ativam o estágio 2.
// Stage-2 agora é acionado para fraudCount ∈ {1,2,3,4} — qualquer resultado
// não-extremo. O modo expand (stage-2 continua stage-1) garante zero trabalho duplicado.
func BenchmarkSearchAmbiguous(b *testing.B) {
	idx := openIndex(b)
	// Vetor intermediário: mistura características legítimas e fraudulentas
	// para ter chance de fraudCount ∈ {2,3}.
	ambiguous := []float32{0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.005}
	// Aquecimento
	for i := 0; i < 5; i++ {
		idx.Search(ambiguous, 5)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(ambiguous, 5)
	}
}

// BenchmarkTopKFraudCountFloat32 mede o custo da seleção float32 de todos os candidatos.
func BenchmarkTopKFraudCountFloat32(b *testing.B) {
	idx := openIndex(b)
	// Query aleatória realista.
	query := make([]float32, 14)
	for i := range query {
		query[i] = float32(i) / 14.0
	}
	// Constrói um pool simulado com efFast=30 candidatos arbitrários.
	h := make([]cand, 30)
	for i := range h {
		h[i] = cand{idx: i * 100, d: int32(i * 50)}
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = idx.topKFraudCountFloat32(query, h, 5)
	}
}
