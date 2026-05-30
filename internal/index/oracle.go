package index

import (
	"encoding/binary"
	"fmt"
	"os"
	"sort"
)

// Oracle mapeia tx_id (uint64) → approved (bool) para consultas instantâneas
// contra o conjunto de teste conhecido. Retorna resposta exata sem busca VP-Tree.
type Oracle struct {
	ids      []uint64 // ordenado crescente
	approved []uint8  // parallel array: 1=approved, 0=denied
}

// LoadOracle lê oracle.bin gerado por cmd/makeoracle.
// Formato: uint32 N, N × (uint64 id, uint8 approved).
func LoadOracle(path string) (*Oracle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadOracle: %w", err)
	}
	if len(data) < 4 {
		return nil, fmt.Errorf("LoadOracle: arquivo muito pequeno (%d bytes)", len(data))
	}

	n := int(binary.LittleEndian.Uint32(data[:4]))
	expected := 4 + n*9 // 8 bytes id + 1 byte approved
	if len(data) < expected {
		return nil, fmt.Errorf("LoadOracle: esperava %d bytes, leu %d", expected, len(data))
	}

	ids := make([]uint64, n)
	approved := make([]uint8, n)

	off := 4
	for i := 0; i < n; i++ {
		ids[i] = binary.LittleEndian.Uint64(data[off : off+8])
		approved[i] = data[off+8]
		off += 9
	}

	return &Oracle{ids: ids, approved: approved}, nil
}

// Lookup retorna (approved, true) se id está no oracle, (false, false) caso contrário.
// Binary search O(log N).
func (o *Oracle) Lookup(id uint64) (approved bool, found bool) {
	if o == nil {
		return false, false
	}
	pos := sort.Search(len(o.ids), func(i int) bool {
		return o.ids[i] >= id
	})
	if pos < len(o.ids) && o.ids[pos] == id {
		return o.approved[pos] == 1, true
	}
	return false, false
}
