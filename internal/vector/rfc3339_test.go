package vector

import (
	"testing"
	"time"
)

func TestParseHourWeekday(t *testing.T) {
	cases := []struct {
		s    string
		hour int
		dow  int // 0=seg..6=dom
		ok   bool
	}{
		// exemplo legítimo do documento (quarta-feira, 2026-03-11, 18h)
		{"2026-03-11T18:45:53Z", 18, 2, true},
		// exemplo fraudulento (sábado, 2026-03-14, 05h)
		{"2026-03-14T05:15:12Z", 5, 5, true},
		// segunda-feira hora 0
		{"2026-03-09T00:00:00Z", 0, 0, true},
		// domingo hora 23
		{"2026-03-15T23:59:59Z", 23, 6, true},
		// string curta inválida
		{"2026-03-11", 0, 0, false},
		// string vazia
		{"", 0, 0, false},
		// com milissegundos
		{"2026-03-11T18:45:53.123Z", 18, 2, true},
	}
	for _, c := range cases {
		h, d, ok := ParseHourWeekday(c.s)
		if ok != c.ok {
			t.Errorf("ParseHourWeekday(%q) ok=%v want %v", c.s, ok, c.ok)
			continue
		}
		if !ok {
			continue
		}
		if h != c.hour {
			t.Errorf("ParseHourWeekday(%q) hour=%d want %d", c.s, h, c.hour)
		}
		if d != c.dow {
			t.Errorf("ParseHourWeekday(%q) dow=%d want %d (0=seg)", c.s, d, c.dow)
		}
	}
}

func TestParseEpochSeconds(t *testing.T) {
	cases := []string{
		"2026-03-11T18:45:53Z",
		"2026-03-14T05:15:12Z",
		"1970-01-01T00:00:00Z",
		"2000-02-29T12:00:00Z", // bissexto
		"2024-02-29T00:00:01Z", // bissexto recente
	}
	for _, s := range cases {
		got, ok := ParseEpochSeconds(s)
		if !ok {
			t.Errorf("ParseEpochSeconds(%q) ok=false", s)
			continue
		}
		// Valida contra time.Parse
		ref, err := time.Parse(time.RFC3339, s)
		if err != nil {
			t.Fatalf("time.Parse(%q): %v", s, err)
		}
		want := ref.Unix()
		if got != want {
			t.Errorf("ParseEpochSeconds(%q) = %d, want %d (diff=%d)", s, got, want, got-want)
		}
	}
}

func TestParseMinutesBetween(t *testing.T) {
	a := "2026-03-11T18:45:00Z"
	b := "2026-03-11T19:15:00Z"
	mins, ok := ParseMinutesBetween(a, b)
	if !ok {
		t.Fatal("ParseMinutesBetween: ok=false")
	}
	if mins < 29.9 || mins > 30.1 {
		t.Errorf("ParseMinutesBetween: got %.2f, want 30.0", mins)
	}
}

func TestVectorizeFraud(t *testing.T) {
	// Exemplo fraudulento do documento REGRAS_DE_DETECCAO.md
	p := Payload{
		Transaction: Transaction{
			Amount:       9505.97,
			Installments: 10,
			RequestedAt:  "2026-03-14T05:15:12Z",
		},
		Customer: Customer{
			AvgAmount:      81.28,
			TxCount24h:     20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: Merchant{
			ID:        "MERC-068",
			MCC:       "7802",
			AvgAmount: 54.86,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  952.27,
		},
		LastTransaction: nil,
	}

	norm := NormalizationConstants{
		MaxAmount: 10000, MaxInstallments: 12,
		AmountVsAvgRatio: 10, MaxMinutes: 1440,
		MaxKm: 1000, MaxTxCount24h: 20,
		MaxMerchantAvgAmount: 10000,
	}
	mccRisk := map[string]float32{"7802": 0.75}

	got := Vectorize(p, norm, mccRisk)

	// Esperado per documento: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
	expected := [14]float32{
		0.9506, 0.8333, 1.0, 0.2174, 0.8333,
		-1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055,
	}
	for i, e := range expected {
		diff := got[i] - e
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("dim[%d]: got %.4f, want %.4f", i, got[i], e)
		}
	}
}

// BenchmarkParseHourWeekday compara nosso parser com time.Parse.
func BenchmarkParseHourWeekday(b *testing.B) {
	s := "2026-03-11T18:45:53Z"
	b.Run("custom", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ParseHourWeekday(s)
		}
	})
	b.Run("time.Parse", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			time.Parse(time.RFC3339, s) //nolint:errcheck
		}
	})
}

// BenchmarkParseEpochSeconds compara nosso parser com time.Parse.Unix().
func BenchmarkParseEpochSeconds(b *testing.B) {
	s := "2026-03-11T18:45:53Z"
	b.Run("custom", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ParseEpochSeconds(s)
		}
	})
	b.Run("time.Parse", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			t, _ := time.Parse(time.RFC3339, s)
			_ = t.Unix()
		}
	})
}
