package vector

// rfc3339.go — parsing de timestamps RFC3339 sem time.Parse nem alocações.
//
// Formato suportado: YYYY-MM-DDTHH:MM:SS[.sss...]Z
// (UTC apenas, sem offset timezone — todos os timestamps do dataset são Z)
//
// Funções exportadas para benchmark/teste:
//   ParseHourWeekday(s string) (hour, dow int, ok bool)
//   ParseEpochSeconds(s string) (int64, bool)

// digit converte um byte ASCII para inteiro; retorna -1 se inválido.
func digit(b byte) int {
	if b >= '0' && b <= '9' {
		return int(b - '0')
	}
	return -1
}

// parse2 lê 2 dígitos em s[off:off+2] e retorna o inteiro.
// Retorna -1 se qualquer dígito for inválido.
func parse2(s string, off int) int {
	if off+2 > len(s) {
		return -1
	}
	h := digit(s[off])
	l := digit(s[off+1])
	if h < 0 || l < 0 {
		return -1
	}
	return h*10 + l
}

// parse4 lê 4 dígitos em s[off:off+4].
func parse4(s string, off int) int {
	if off+4 > len(s) {
		return -1
	}
	return digit(s[off])*1000 + digit(s[off+1])*100 + digit(s[off+2])*10 + digit(s[off+3])
}

// ParseHourWeekday extrai hora UTC (0-23) e dia da semana (seg=0..dom=6)
// de uma string RFC3339 UTC, sem alocar um time.Time.
//
// Formato esperado: "YYYY-MM-DDTHH:MM:SS[.frac]Z"
// Retorna ok=false se o formato for inválido (vetor recebe zero para essas dims).
func ParseHourWeekday(s string) (hour, dow int, ok bool) {
	if len(s) < 20 {
		return 0, 0, false
	}
	year := parse4(s, 0)
	month := parse2(s, 5)
	day := parse2(s, 8)
	h := parse2(s, 11)
	if year < 0 || month < 0 || day < 0 || h < 0 {
		return 0, 0, false
	}
	if month < 1 || month > 12 || day < 1 || day > 31 || h < 0 || h > 23 {
		return 0, 0, false
	}

	// Dia da semana via algoritmo de Tomohiko Sakamoto (resultado: 0=dom..6=sab).
	// Convertemos para seg=0..dom=6 conforme especificado em REGRAS_DE_DETECCAO.md.
	wd := weekdayTomohiko(year, month, day) // 0=dom
	dow = (wd + 6) % 7                      // 0=seg..6=dom

	return h, dow, true
}

// weekdayTomohiko retorna 0=domingo..6=sábado usando o algoritmo de Tomohiko Sakamoto.
// Fonte: https://en.wikipedia.org/wiki/Determination_of_the_day_of_the_week#Sakamoto's_methods
func weekdayTomohiko(year, month, day int) int {
	t := [12]int{0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4}
	if month < 3 {
		year--
	}
	return (year + year/4 - year/100 + year/400 + t[month-1] + day) % 7
}

// isLeap indica se o ano é bissexto.
func isLeap(year int) bool {
	return year%400 == 0 || (year%4 == 0 && year%100 != 0)
}

// cumDaysNonLeap[m] = dias acumulados de 1-Jan até início do mês m (ano não-bissexto).
// cumDaysLeap[m]    = idem para ano bissexto.
var cumDaysNonLeap = [13]int64{0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334}
var cumDaysLeap = [13]int64{0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335}

// ParseEpochSeconds converte uma string RFC3339 UTC para segundos desde Unix epoch.
// Não aloca. Retorna ok=false se o formato for inválido.
//
// Fórmula O(1) para dias desde 1970-01-01 até 1-Jan do ano dado:
//   y1 = year-1
//   yearDays = 365*(year-1970) + (y1/4 - y1/100 + y1/400) - (1969/4 - 1969/100 + 1969/400)
//            = 365*(year-1970) + y1/4 - y1/100 + y1/400 - 477
// Deriva-se de: dias do calendário gregoriano = 365*year + year/4 - year/100 + year/400
// descontando o valor para 1969 (= 477 anos bissextos de 1 AD a 1969).
func ParseEpochSeconds(s string) (int64, bool) {
	if len(s) < 20 {
		return 0, false
	}
	year := parse4(s, 0)
	month := parse2(s, 5)
	day := parse2(s, 8)
	hour := parse2(s, 11)
	min := parse2(s, 14)
	sec := parse2(s, 17)
	if year < 0 || month < 1 || month > 12 || day < 1 || day > 31 ||
		hour < 0 || hour > 23 || min < 0 || min > 59 || sec < 0 || sec > 60 {
		return 0, false
	}

	// Dias de 1970-01-01 até 1-Jan do ano dado (fórmula O(1), sem loop).
	y1 := int64(year - 1)
	yearDays := 365*int64(year-1970) + y1/4 - y1/100 + y1/400 - 477

	// Dias dos meses completos do ano atual (lookup O(1)).
	var monthDays int64
	if isLeap(year) {
		monthDays = cumDaysLeap[month]
	} else {
		monthDays = cumDaysNonLeap[month]
	}

	days := yearDays + monthDays + int64(day-1)
	epoch := days*86400 + int64(hour)*3600 + int64(min)*60 + int64(sec)
	return epoch, true
}

// ParseMinutesBetween calcula os minutos entre dois timestamps RFC3339 UTC
// (b - a, pode ser negativo). Não aloca.
// Retorna ok=false se qualquer string for inválida.
func ParseMinutesBetween(a, b string) (minutes float32, ok bool) {
	ea, oka := ParseEpochSeconds(a)
	eb, okb := ParseEpochSeconds(b)
	if !oka || !okb {
		return 0, false
	}
	diff := eb - ea
	return float32(diff) / 60.0, true
}
