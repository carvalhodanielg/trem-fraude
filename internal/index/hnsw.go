package index

import (
	"compress/gzip"
	"encoding/json"
	"log"
	"math"
	"os"
	"time"
)

const dims = 14

// sentinela para last_transaction null (original = -1.0)
const sentinelValue = uint8(254)

type Index struct {
	vectors []uint8 // quantizado: 1 byte por dimensão em vez de 4
	labels  []uint8
	count   int
}

type refEntry struct {
	Vector [14]float32 `json:"vector"`
	Label  string      `json:"label"`
}

func quantize(v float32) uint8 {
	if v < 0 {
		return sentinelValue // -1.0 sentinel
	}
	if v >= 1.0 {
		return 253
	}
	return uint8(v * 253)
}

func Load(path string) (*Index, error) {
	start := time.Now()
	log.Println("carregando vetores...")

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	idx := &Index{
		vectors: make([]uint8, 0, 3_000_000*dims),
		labels:  make([]uint8, 0, 3_000_000),
	}

	dec := json.NewDecoder(gz)
	dec.Token()

	count := 0
	for dec.More() {
		var entry refEntry
		if err := dec.Decode(&entry); err != nil {
			return nil, err
		}

		for _, f := range entry.Vector {
			idx.vectors = append(idx.vectors, quantize(f))
		}

		if entry.Label == "fraud" {
			idx.labels = append(idx.labels, 1)
		} else {
			idx.labels = append(idx.labels, 0)
		}

		count++
		if count%500000 == 0 {
			log.Printf("  carregados %d vetores...", count)
		}
	}

	idx.count = count
	log.Printf("carregamento pronto: %d vetores em %s", count, time.Since(start))
	return idx, nil
}

func euclidean(a []uint8, b []uint8) float32 {
	var sum float32
	for i := 0; i < dims; i++ {
		// sentinela: se qualquer um dos dois é -1, distância nessa dimensão = 0
		if a[i] == sentinelValue || b[i] == sentinelValue {
			if a[i] == b[i] {
				continue // ambos null — sem distância
			}
			// um tem, outro não — penalidade fixa
			sum += 1.0
			continue
		}
		d := float32(int16(a[i]) - int16(b[i]))
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

type neighbor struct {
	dist  float32
	label uint8
}

func (idx *Index) Search(query []float32, k int) float32 {
	// quantiza a query uma vez antes do loop
	qVec := make([]uint8, dims)
	for i, f := range query {
		qVec[i] = quantize(f)
	}

	worst := float32(math.MaxFloat32)
	neighbors := make([]neighbor, 0, k)

	for i := 0; i < idx.count; i++ {
		vec := idx.vectors[i*dims : i*dims+dims]
		d := euclidean(qVec, vec)

		if len(neighbors) < k {
			neighbors = append(neighbors, neighbor{d, idx.labels[i]})
			if len(neighbors) == k {
				worst = 0
				for _, n := range neighbors {
					if n.dist > worst {
						worst = n.dist
					}
				}
			}
			continue
		}

		if d < worst {
			worstIdx := 0
			for j, n := range neighbors {
				if n.dist > neighbors[worstIdx].dist {
					worstIdx = j
				}
			}
			neighbors[worstIdx] = neighbor{d, idx.labels[i]}
			worst = 0
			for _, n := range neighbors {
				if n.dist > worst {
					worst = n.dist
				}
			}
		}
	}

	var fraudCount float32
	for _, n := range neighbors {
		if n.label == 1 {
			fraudCount++
		}
	}
	return fraudCount / float32(k)
}
