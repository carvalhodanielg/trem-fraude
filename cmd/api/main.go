package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sync/atomic"

	"github.com/carvalhodanielg/trem-de-fraude/internal/index"
	"github.com/carvalhodanielg/trem-de-fraude/internal/vector"
)

var (
	idxPtr  atomic.Pointer[index.Index]
	normPtr atomic.Pointer[vector.NormalizationConstants]
	riskPtr atomic.Pointer[map[string]float32]
)

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
	http.HandleFunc("GET /ready", handleReady)
	http.HandleFunc("POST /fraud-score", handleFraudScore)

	go startup()

	log.Println("listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
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

	idx, err := index.Load("resources/references.json.gz")
	if err != nil {
		log.Fatalf("erro ao carregar índice: %v", err)
	}

	// idx é o último a ser armazenado — é ele que sinaliza que está pronto
	idxPtr.Store(idx)
	log.Println("pronto para receber requisições")
}

func handleReady(w http.ResponseWriter, r *http.Request) {
	if idxPtr.Load() != nil {
		w.WriteHeader(http.StatusOK)
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
}

func handleFraudScore(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			log.Printf("panic recuperado: %v", rec)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(fraudResponse{Approved: true, FraudScore: 0.0})
		}
	}()

	idx := idxPtr.Load()
	norm := normPtr.Load()
	risk := riskPtr.Load()

	if idx == nil || norm == nil || risk == nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(fraudResponse{Approved: true, FraudScore: 0.0})
		return
	}

	var req fraudRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(fraudResponse{Approved: true, FraudScore: 0.0})
		return
	}

	payload := vector.Payload{
		Transaction:     req.Transaction,
		Customer:        req.Customer,
		Merchant:        req.Merchant,
		Terminal:        req.Terminal,
		LastTransaction: req.LastTransaction,
	}

	vec := vector.Vectorize(payload, *norm, *risk)
	score := idx.Search(vec[:], 5)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(fraudResponse{
		Approved:   score < 0.6,
		FraudScore: score,
	})
}
