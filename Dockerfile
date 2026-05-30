FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# -pgo=auto usa cmd/api/default.pgo para Profile-Guided Optimization (Go 1.21+).
# O profile foi coletado sob carga real (374k req/30s); ganho típico: 5-15% no p99.
RUN go build -pgo=auto -ldflags="-s -w" -o api ./cmd/api
RUN go build -ldflags="-s -w" -o preprocess ./cmd/preprocess
# Constrói o índice HNSW + VP-Tree a partir de references.json.gz (~3-4 min, ~900 MB RAM)
# Gera: resources/index.bin (vetores+labels+HNSW) e resources/vptree.bin (VP-Tree)
RUN ./preprocess

FROM alpine:3.19
WORKDIR /app
COPY --from=builder /app/api .
COPY --from=builder /app/resources/index.bin ./resources/
COPY --from=builder /app/resources/vptree.bin ./resources/
COPY resources/mcc_risk.json ./resources/
COPY resources/normalization.json ./resources/
EXPOSE 8080
CMD ["./api"]
