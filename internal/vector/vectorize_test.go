package vector

import (
	"encoding/json"
	"testing"
)

func TestVectorizeLegit(t *testing.T) {
	raw := `{
        "id": "tx-1329056812",
        "transaction":      { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
        "customer":         { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
        "merchant":         { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
        "terminal":         { "is_online": false, "card_present": true, "km_from_home": 29.23 },
        "last_transaction": null
    }`

	var p Payload
	json.Unmarshal([]byte(raw), &p)

	norm := NormalizationConstants{
		MaxAmount: 10000, MaxInstallments: 12,
		AmountVsAvgRatio: 10, MaxMinutes: 1440,
		MaxKm: 1000, MaxTxCount24h: 20,
		MaxMerchantAvgAmount: 10000,
	}

	mccRisk := map[string]float32{"5411": 0.15}

	got := Vectorize(p, norm, mccRisk)

	// esperado conforme o documento
	expected := [14]float32{
		0.0041, 0.1667, 0.05, 0.7826, 0.3333,
		-1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006,
	}

	for i, e := range expected {
		diff := got[i] - e
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("dim[%d]: got %.4f, want %.4f", i, got[i], e)
		}
	}
}
