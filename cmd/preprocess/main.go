// Preprocess constrói o índice HNSW compacto a partir de references.json.gz.
//
// Formato do binário de saída (index.bin):
//
//	[4] uint32 N          — total de nós
//	[4] uint32 M          — max vizinhos por nó (stride = 2*M)
//	[4] uint32 nLayers    — número de camadas HNSW
//	[4] uint32 entryIdx   — nó de entrada (camada mais alta)
//	[N*14] int8           — vetores quantizados (flat, stride=14)
//	[N] uint8             — labels (0=legit, 1=fraud)
//	--- Camada 0 (todos os N nós, stride fixo) ---
//	[N * stride * 3] byte — adj0: vizinhos em uint24 LE (stride slots por nó, 0-padded)
//	[N] uint8             — adj0Counts: vizinhos válidos por nó
//	--- Camadas 1..nLayers-1 (CSR por camada) ---
//	  [4] uint32 Nl          — nós nessa camada
//	  [Nl] uint32            — nodeIdxs (ordenados crescente)
//	  [Nl+1] uint32          — offsets CSR
//	  [offsets[Nl]] uint32   — vizinhos
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"io"
	"log"
	"math"
	"os"
	"sort"
	"time"

	"github.com/coder/hnsw"
)

const dim = 14

type refEntry struct {
	Vector [dim]float32 `json:"vector"`
	Label  string       `json:"label"`
}

// quantize converte float32 → int8.
// Valores em [0,1] → [0,127]. Sentinela −1.0 → int8(−1).
func quantize(v float32) int8 {
	if v < -0.5 { // único caso negativo: sentinela −1.0
		return -1
	}
	if v < 0 {
		return 0
	}
	r := math.Round(float64(v) * 127)
	if r > 127 {
		r = 127
	}
	return int8(r)
}

type parsedNode struct {
	idx       int
	neighbors []int // índices (não keys)
}

type parsedLayer struct {
	nodes []parsedNode // ordenados por idx crescente
}

func main() {
	input := "resources/references.json.gz"
	output := "resources/index.bin"
	if len(os.Args) > 1 {
		input = os.Args[1]
	}
	if len(os.Args) > 2 {
		output = os.Args[2]
	}

	// ── Fase 1: Ler vetores e construir grafo HNSW ─────────────────────────
	log.Printf("fase 1: lendo %s e construindo grafo HNSW...", input)
	start := time.Now()

	f, err := os.Open(input)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}

	graph := hnsw.NewGraph[int]()
	graph.M = 4         // adj0 uint24: 72 MB — mais conectividade, menos mínimos locais; file-backed mmap não conta contra limite do container
	graph.EfSearch = 200 // alta qualidade na construção
	graph.Distance = hnsw.EuclideanDistance

	dec := json.NewDecoder(gz)
	dec.Token() // skip '['

	rawVectors := make([][dim]float32, 0, 3_000_000)
	rawLabels := make([]uint8, 0, 3_000_000)

	count := 0
	for dec.More() {
		var entry refEntry
		if err := dec.Decode(&entry); err != nil {
			log.Fatal(err)
		}

		// key par = legit, ímpar = fraud
		nodeID := count * 2
		var label uint8
		if entry.Label == "fraud" {
			nodeID = count*2 + 1
			label = 1
		}
		rawLabels = append(rawLabels, label)
		rawVectors = append(rawVectors, entry.Vector)

		graph.Add(hnsw.MakeNode(nodeID, entry.Vector[:]))
		count++

		if count%500_000 == 0 {
			log.Printf("  %d nós em %s", count, time.Since(start))
		}
	}
	gz.Close()

	N := count
	M := graph.M
	stride := 2 * M
	log.Printf("grafo construído: %d nós, M=%d, stride=%d em %s", N, M, stride, time.Since(start))

	// ── Fase 2: Export da lib → arquivo temporário → parse ────────────────
	log.Println("fase 2: exportando grafo para parse da estrutura...")

	tmpFile, err := os.CreateTemp("", "hnsw-export-*.bin")
	if err != nil {
		log.Fatal(err)
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	{
		bw := bufio.NewWriterSize(tmpFile, 8<<20)
		if err := graph.Export(bw); err != nil {
			log.Fatal("export:", err)
		}
		if err := bw.Flush(); err != nil {
			log.Fatal("flush:", err)
		}
		tmpFile.Close()
	}

	graph = nil // libera ~RAM usada pelos maps internos da lib
	log.Println("fase 2: parseando estrutura do grafo exportado...")

	tf, err := os.Open(tmpPath)
	if err != nil {
		log.Fatal(err)
	}
	br := bufio.NewReaderSize(tf, 8<<20)

	// ── Leitores do formato binário da biblioteca ──────────────────────────
	// int: varint (binary.ReadVarint)
	// float64: 8 bytes LE
	// string: varint(len) + bytes
	// []float32: varint(len) + len*4 bytes LE

	readVarint := func() int {
		v, err := binary.ReadVarint(br)
		if err != nil {
			log.Fatal("readVarint:", err)
		}
		return int(v)
	}
	readFloat64 := func() float64 {
		var v float64
		if err := binary.Read(br, binary.LittleEndian, &v); err != nil {
			log.Fatal("readFloat64:", err)
		}
		return v
	}
	readString := func() string {
		ln := readVarint()
		b := make([]byte, ln)
		if _, err := io.ReadFull(br, b); err != nil {
			log.Fatal("readString:", err)
		}
		return string(b)
	}
	skipFloat32Slice := func() {
		ln := readVarint()
		// ln floats × 4 bytes — descartamos pois usamos rawVectors
		buf := make([]byte, ln*4)
		if _, err := io.ReadFull(br, buf); err != nil {
			log.Fatal("skipFloat32Slice:", err)
		}
	}

	// ── Cabeçalho do export da biblioteca ─────────────────────────────────
	version := readVarint()
	if version != 1 {
		log.Fatalf("versão de encoding desconhecida: %d", version)
	}
	exportM := readVarint()
	exportMl := readFloat64()
	exportEf := readVarint()
	distName := readString()
	log.Printf("  export: M=%d Ml=%.3f EfSearch=%d dist=%s", exportM, exportMl, exportEf, distName)

	nLayers := readVarint()
	log.Printf("  %d camadas no grafo", nLayers)

	layers := make([]parsedLayer, nLayers)

	for l := 0; l < nLayers; l++ {
		nNodes := readVarint()
		pl := parsedLayer{nodes: make([]parsedNode, 0, nNodes)}

		for j := 0; j < nNodes; j++ {
			key := readVarint()
			skipFloat32Slice()         // vetor já em rawVectors
			nNeighbors := readVarint()
			neighbors := make([]int, nNeighbors)
			for k := 0; k < nNeighbors; k++ {
				nKey := readVarint()
				neighbors[k] = nKey / 2 // key → índice
			}
			pl.nodes = append(pl.nodes, parsedNode{
				idx:       key / 2,
				neighbors: neighbors,
			})
		}

		// Ordena por idx para output determinístico
		sort.Slice(pl.nodes, func(a, b int) bool {
			return pl.nodes[a].idx < pl.nodes[b].idx
		})
		layers[l] = pl
		log.Printf("  camada %d: %d nós parseados", l, nNodes)
	}
	tf.Close()

	// Entry point: qualquer nó da camada mais alta (topo)
	entryIdx := layers[nLayers-1].nodes[0].idx
	log.Printf("  entry point: nó %d", entryIdx)

	// ── Fase 3: Escrever binário compacto ─────────────────────────────────
	log.Printf("fase 3: escrevendo %s...", output)

	outFile, err := os.Create(output)
	if err != nil {
		log.Fatal(err)
	}
	bw := bufio.NewWriterSize(outFile, 8<<20)

	write32 := func(v uint32) {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], v)
		bw.Write(buf[:])
	}

	// write24 escreve um uint32 em 3 bytes LE (uint24). N < 2^24 garante que não há overflow.
	write24 := func(v uint32) {
		bw.WriteByte(byte(v))
		bw.WriteByte(byte(v >> 8))
		bw.WriteByte(byte(v >> 16))
	}

	// Cabeçalho
	write32(uint32(N))
	write32(uint32(M))
	write32(uint32(nLayers))
	write32(uint32(entryIdx))

	// Vetores quantizados (N × dim × int8)
	for _, vec := range rawVectors {
		for d := 0; d < dim; d++ {
			bw.WriteByte(byte(quantize(vec[d])))
		}
	}
	// rawVectors mantido vivo — será usado na fase 4 (VP-Tree)
	log.Println("  vetores escritos")

	// Labels (N × uint8)
	bw.Write(rawLabels)
	rawLabels = nil
	log.Println("  labels escritos")

	// ── Camada 0: stride fixo ──────────────────────────────────────────────
	// adj0[i*stride : i*stride+stride] = vizinhos do nó i (0-padded)
	// adj0Counts[i] = quantidade real de vizinhos
	adj0 := make([]uint32, N*stride)
	adj0Counts := make([]uint8, N)

	for _, pn := range layers[0].nodes {
		i := pn.idx
		cnt := len(pn.neighbors)
		if cnt > stride {
			cnt = stride
		}
		adj0Counts[i] = uint8(cnt)
		base := i * stride
		for j, nbr := range pn.neighbors[:cnt] {
			adj0[base+j] = uint32(nbr)
		}
	}

	// Escreve adj0 em formato uint24 LE (3 bytes por vizinho)
	// para economizar 24 MB vs uint32 com M=4, mantendo adj0 em ~72 MB.
	for _, v := range adj0 {
		write24(v)
	}
	adj0 = nil
	bw.Write(adj0Counts)
	adj0Counts = nil
	log.Printf("  camada 0 escrita (%d nós, stride=%d, uint24)", N, stride)

	// ── Camadas superiores (l=1..nLayers-1): CSR ──────────────────────────
	for l := 1; l < nLayers; l++ {
		pl := layers[l]
		Nl := len(pl.nodes)
		write32(uint32(Nl))

		for _, pn := range pl.nodes {
			write32(uint32(pn.idx))
		}

		offset := uint32(0)
		write32(offset)
		for _, pn := range pl.nodes {
			offset += uint32(len(pn.neighbors))
			write32(offset)
		}

		totalEdges := 0
		for _, pn := range pl.nodes {
			for _, nbr := range pn.neighbors {
				write32(uint32(nbr))
			}
			totalEdges += len(pn.neighbors)
		}
		log.Printf("  camada %d escrita: %d nós, %d arestas", l, Nl, totalEdges)
	}

	if err := bw.Flush(); err != nil {
		log.Fatal(err)
	}
	if err := outFile.Close(); err != nil {
		log.Fatal(err)
	}

	stat, _ := os.Stat(output)
	log.Printf("índice HNSW salvo: %s (%.1f MB)", output, float64(stat.Size())/1024/1024)

	// ── Fase 4: Construir VP-Tree e escrever vptree.bin ───────────────────
	log.Println("fase 4: construindo VP-Tree...")
	vpOutput := "resources/vptree.bin"
	if len(os.Args) > 3 {
		vpOutput = os.Args[3]
	}

	vpNodes, vpPerm := buildVPTree(rawVectors)
	rawVectors = nil // libera memória após o build
	log.Printf("VP-Tree construída: %d nós, %d pontos", len(vpNodes), N)

	if err := writeVPTree(vpOutput, N, vpNodes, vpPerm); err != nil {
		log.Fatal("writeVPTree:", err)
	}

	log.Printf("concluído — tempo total: %s", time.Since(start))
}
