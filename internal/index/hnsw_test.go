package index

import (
	"testing"
	"time"
)

func TestLoad(t *testing.T) {
	idx, err := Load("../../resources/references.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	if idx == nil {
		t.Fatal("índice nulo")
	}
}

func TestSearch(t *testing.T) {
	idx, err := Load("../../resources/references.json.gz")
	if err != nil {
		t.Fatal(err)
	}

	// transação fraudulenta do documento — esperado fraud_score = 1.0
	fraud := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	score := idx.Search(fraud, 5)
	t.Logf("fraud score (esperado ~1.0): %.2f", score)
	if score < 0.6 {
		t.Errorf("esperado score alto, got %.2f", score)
	}

	// transação legítima do documento — esperado fraud_score = 0.0
	legit := []float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	score = idx.Search(legit, 5)
	t.Logf("legit score (esperado ~0.0): %.2f", score)
	if score > 0.4 {
		t.Errorf("esperado score baixo, got %.2f", score)
	}
}

func TestSearchLatency(t *testing.T) {
	idx, err := Load("../../resources/references.json.gz")
	if err != nil {
		t.Fatal(err)
	}

	query := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}

	start := time.Now()
	iterations := 10
	for i := 0; i < iterations; i++ {
		idx.Search(query, 5)
	}
	elapsed := time.Since(start)

	t.Logf("média por busca: %s", elapsed/time.Duration(iterations))
	t.Logf("total %d buscas: %s", iterations, elapsed)
}
