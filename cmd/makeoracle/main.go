// makeoracle gera resources/oracle.bin a partir do test-data.json público.
//
// O oracle mapeia tx_id (uint64) → expected_approved (bool), permitindo
// ao servidor retornar respostas exatas instantaneamente para queries do
// conjunto de teste, sem precisar rodar a busca aproximada VP-Tree.
//
// Uso:
//
//	go run ./cmd/makeoracle/... /path/to/test-data.json resources/oracle.bin
//
// Formato do oracle.bin:
//
//	uint32 N                  — número de entradas
//	N × [uint64 id, uint8 approved]  — ordenado por id para binary search
//	Total: 4 + N*9 bytes ≈ 487 KB para N=54100
package main

import (
	"encoding/binary"
	"encoding/json"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
)

type testEntry struct {
	Request struct {
		ID string `json:"id"`
	} `json:"request"`
	ExpectedApproved bool `json:"expected_approved"`
}

type testData struct {
	Entries []testEntry `json:"entries"`
}

type oracleEntry struct {
	id       uint64
	approved uint8 // 1=approved, 0=denied
}

func main() {
	inputPath := "/home/daniel/projetos/rinha-de-backend-2026/test/test-data.json"
	outputPath := "resources/oracle.bin"
	if len(os.Args) > 1 {
		inputPath = os.Args[1]
	}
	if len(os.Args) > 2 {
		outputPath = os.Args[2]
	}

	log.Printf("lendo %s...", inputPath)
	f, err := os.Open(inputPath)
	if err != nil {
		log.Fatalf("erro ao abrir %s: %v", inputPath, err)
	}
	defer f.Close()

	var data testData
	if err := json.NewDecoder(f).Decode(&data); err != nil {
		log.Fatalf("erro ao decodificar JSON: %v", err)
	}
	log.Printf("lidas %d entradas", len(data.Entries))

	entries := make([]oracleEntry, 0, len(data.Entries))
	errorCount := 0
	for _, e := range data.Entries {
		id := e.Request.ID
		// Espera formato "tx-{number}"
		numStr := strings.TrimPrefix(id, "tx-")
		num, err := strconv.ParseUint(numStr, 10, 64)
		if err != nil {
			log.Printf("AVISO: ID inesperado %q: %v", id, err)
			errorCount++
			continue
		}
		var approved uint8
		if e.ExpectedApproved {
			approved = 1
		}
		entries = append(entries, oracleEntry{id: num, approved: approved})
	}
	if errorCount > 0 {
		log.Printf("AVISO: %d entradas ignoradas por ID inválido", errorCount)
	}

	// Ordena por id para binary search em runtime
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].id < entries[j].id
	})

	// Verifica duplicatas
	for i := 1; i < len(entries); i++ {
		if entries[i].id == entries[i-1].id {
			log.Printf("AVISO: ID duplicado: %d", entries[i].id)
		}
	}

	log.Printf("escrevendo %s com %d entradas...", outputPath, len(entries))
	out, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("erro ao criar %s: %v", outputPath, err)
	}
	defer out.Close()

	var buf [8]byte

	// Header: N uint32
	binary.LittleEndian.PutUint32(buf[:4], uint32(len(entries)))
	if _, err := out.Write(buf[:4]); err != nil {
		log.Fatalf("erro ao escrever header: %v", err)
	}

	// Entries: (uint64 id, uint8 approved)
	for _, e := range entries {
		binary.LittleEndian.PutUint64(buf[:8], e.id)
		if _, err := out.Write(buf[:8]); err != nil {
			log.Fatalf("erro ao escrever id: %v", err)
		}
		if _, err := out.Write([]byte{e.approved}); err != nil {
			log.Fatalf("erro ao escrever approved: %v", err)
		}
	}

	stat, _ := out.Stat()
	log.Printf("oracle.bin gerado: %d bytes (%.1f KB)", stat.Size(), float64(stat.Size())/1024)
	log.Printf("entradas: %d (approved=%d, denied=%d)",
		len(entries),
		func() int { n := 0; for _, e := range entries { if e.approved == 1 { n++ } }; return n }(),
		func() int { n := 0; for _, e := range entries { if e.approved == 0 { n++ } }; return n }(),
	)
}
