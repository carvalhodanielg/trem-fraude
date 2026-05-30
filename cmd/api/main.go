package main

import (
	"bytes"
	"log"
	"net/http"
	_ "net/http/pprof" // registra /debug/pprof/* para coleta de CPU profile (PGO)
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	gojson "github.com/goccy/go-json"

	"github.com/carvalhodanielg/trem-de-fraude/internal/index"
	"github.com/carvalhodanielg/trem-de-fraude/internal/vector"
)

var (
	idxPtr  atomic.Pointer[index.VPIndex]
	normPtr atomic.Pointer[vector.NormalizationConstants]
	riskPtr atomic.Pointer[map[string]float32]
)

// reqBufPool reutiliza bytes.Buffer entre requisições para evitar alocações
// de heap no body reading. Payloads do dataset cabem em < 1 KB.
var reqBufPool = sync.Pool{New: func() any { return new(bytes.Buffer) }}

type fraudRequest struct {
	ID              string                  `json:"id"`
	Transaction     vector.Transaction      `json:"transaction"`
	Customer        vector.Customer         `json:"customer"`
	Merchant        vector.Merchant         `json:"merchant"`
	Terminal        vector.Terminal         `json:"terminal"`
	LastTransaction *vector.LastTransaction `json:"last_transaction"`
}

type fraudResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float32 `json:"fraud_score"`
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", handleReady)
	mux.HandleFunc("POST /fraud-score", handleFraudScore)

	go startup()

	// Servidor de pprof no :6060 (isolado do path de produção).
	// Usado para coleta de CPU profile para PGO:
	//   curl -o cmd/api/default.pgo "http://localhost:6060/debug/pprof/profile?seconds=30"
	// O import de _ "net/http/pprof" registra os handlers no DefaultServeMux.
	go func() {
		log.Println("pprof listening on :6060")
		if err := http.ListenAndServe(":6060", nil); err != nil {
			log.Printf("pprof server error: %v", err)
		}
	}()

	srv := &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Println("listening on :8080")
	log.Fatal(srv.ListenAndServe())
}

func startup() {
	norm, err := vector.LoadNormalization("resources/normalization.json")
	if err != nil {
		log.Fatalf("erro ao carregar normalization.json: %v", err)
	}
	normPtr.Store(&norm)

	mccRisk, err := vector.LoadMCCRisk("resources/mcc_risk.json")
	if err != nil {
		log.Fatalf("erro ao carregar mcc_risk.json: %v", err)
	}
	riskPtr.Store(&mccRisk)

	idx, err := index.LoadVP("resources/index.bin", "resources/vptree.bin")
	if err != nil {
		log.Fatalf("erro ao carregar índice VP-Tree: %v", err)
	}

	// idx é o último a ser armazenado — é ele que sinaliza que está pronto
	idxPtr.Store(idx)

	// Devolve ao OS qualquer heap Go liberado durante o loading.
	// Com mmap, o heap permanente é apenas pools + runtime (~5-15 MB);
	// isso reduz RSS imediatamente, garantindo espaço para allocações do HTTP.
	runtime.GC()
	debug.FreeOSMemory()

	log.Println("pronto para receber requisições")
}

func handleReady(w http.ResponseWriter, r *http.Request) {
	if idxPtr.Load() != nil {
		w.WriteHeader(http.StatusOK)
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
}

var approvedJSON = []byte(`{"approved":true,"fraud_score":0}` + "\n")
var rejectedJSON = []byte(`{"approved":false,"fraud_score":1}` + "\n")

// writeFraudResponse escreve a resposta JSON sem alocações usando strconv.
// Formato: {"approved":true|false,"fraud_score":X.XXXXXX}
func writeFraudResponse(w http.ResponseWriter, approved bool, score float32) {
	var buf [64]byte
	b := buf[:0]
	b = append(b, `{"approved":`...)
	if approved {
		b = append(b, "true"...)
	} else {
		b = append(b, "false"...)
	}
	b = append(b, `,"fraud_score":`...)
	b = strconv.AppendFloat(b, float64(score), 'f', 6, 32)
	b = append(b, '}', '\n')
	w.Write(b) //nolint:errcheck
}

func handleFraudScore(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			log.Printf("panic recuperado: %v", rec)
			w.Header().Set("Content-Type", "application/json")
			w.Write(approvedJSON) //nolint:errcheck
		}
	}()

	idx := idxPtr.Load()
	norm := normPtr.Load()
	risk := riskPtr.Load()

	if idx == nil || norm == nil || risk == nil {
		w.Header().Set("Content-Type", "application/json")
		w.Write(approvedJSON) //nolint:errcheck
		return
	}

	// Lê body em buffer poolado (evita alocação do bytes.Buffer a cada request).
	// go-json é 5-10× mais rápido que encoding/json e usa as interfaces
	// json.Unmarshaler geradas pelo easyjson nos tipos internos (vector.*).
	bodyBuf := reqBufPool.Get().(*bytes.Buffer)
	bodyBuf.Reset()
	bodyBuf.ReadFrom(r.Body) //nolint:errcheck
	var req fraudRequest
	if err := gojson.Unmarshal(bodyBuf.Bytes(), &req); err != nil {
		reqBufPool.Put(bodyBuf)
		w.Header().Set("Content-Type", "application/json")
		w.Write(approvedJSON) //nolint:errcheck
		return
	}
	reqBufPool.Put(bodyBuf)

	payload := vector.Payload{
		Transaction:     req.Transaction,
		Customer:        req.Customer,
		Merchant:        req.Merchant,
		Terminal:        req.Terminal,
		LastTransaction: req.LastTransaction,
	}

	vec := vector.Vectorize(payload, *norm, *risk)
	score := idx.Search(vec[:], 5)

	approved := score < 0.6

	w.Header().Set("Content-Type", "application/json")

	// Fast paths para os casos mais comuns.
	if score == 0 {
		w.Write(approvedJSON) //nolint:errcheck
		return
	}
	if score == 1 {
		w.Write(rejectedJSON) //nolint:errcheck
		return
	}

	// Resposta sem alocações de encoder para scores intermediários.
	writeFraudResponse(w, approved, score)
}
