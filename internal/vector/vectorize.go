package vector

import (
	"encoding/json"
	"os"
	"time"
)

type NormalizationConstants struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKm                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

type Transaction struct {
	Amount       float32 `json:"amount"`
	Installments float32 `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float32  `json:"avg_amount"`
	TxCount24h     float32  `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float32 `json:"km_from_home"`
}

type LastTransaction struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float32 `json:"km_from_current"`
}

type Payload struct {
	Transaction     Transaction      `json:"transaction"`
	Customer        Customer         `json:"customer"`
	Merchant        Merchant         `json:"merchant"`
	Terminal        Terminal         `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"`
}

func LoadNormalization(path string) (NormalizationConstants, error) {
	var norm NormalizationConstants
	data, err := os.ReadFile(path)
	if err != nil {
		return norm, err
	}
	err = json.Unmarshal(data, &norm)
	return norm, err
}

func LoadMCCRisk(path string) (map[string]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var risk map[string]float32
	err = json.Unmarshal(data, &risk)
	return risk, err
}

func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func Vectorize(p Payload, norm NormalizationConstants, mccRisk map[string]float32) [14]float32 {
	var v [14]float32

	v[0] = clamp(p.Transaction.Amount / norm.MaxAmount)
	v[1] = clamp(p.Transaction.Installments / norm.MaxInstallments)
	// Guard contra divisão por zero: AvgAmount==0 geraria NaN que corromperia o vetor.
	amtRatio := float32(0)
	if p.Customer.AvgAmount > 0 {
		amtRatio = (p.Transaction.Amount / p.Customer.AvgAmount) / norm.AmountVsAvgRatio
	}
	v[2] = clamp(amtRatio)

	t, _ := time.Parse(time.RFC3339, p.Transaction.RequestedAt)
	v[3] = float32(t.UTC().Hour()) / 23.0
	dow := (int(t.UTC().Weekday()) + 6) % 7
	v[4] = float32(dow) / 6.0

	if p.LastTransaction == nil {
		v[5] = -1
		v[6] = -1
	} else {
		lastT, _ := time.Parse(time.RFC3339, p.LastTransaction.Timestamp)
		minutes := float32(t.Sub(lastT).Minutes())
		v[5] = clamp(minutes / norm.MaxMinutes)
		v[6] = clamp(p.LastTransaction.KmFromCurrent / norm.MaxKm)
	}

	v[7] = clamp(p.Terminal.KmFromHome / norm.MaxKm)
	v[8] = clamp(p.Customer.TxCount24h / norm.MaxTxCount24h)

	if p.Terminal.IsOnline {
		v[9] = 1
	}
	if p.Terminal.CardPresent {
		v[10] = 1
	}

	known := false
	for _, m := range p.Customer.KnownMerchants {
		if m == p.Merchant.ID {
			known = true
			break
		}
	}
	if !known {
		v[11] = 1
	}

	if risk, ok := mccRisk[p.Merchant.MCC]; ok {
		v[12] = risk
	} else {
		v[12] = 0.5
	}

	v[13] = clamp(p.Merchant.AvgAmount / norm.MaxMerchantAvgAmount)

	return v
}
