FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -ldflags="-s -w" -o api ./cmd/api
RUN go build -ldflags="-s -w" -o preprocess ./cmd/preprocess
# Constrói o índice HNSW a partir de references.json.gz (~3-4 min, ~900 MB RAM)
RUN ./preprocess

FROM alpine:3.19
WORKDIR /app
COPY --from=builder /app/api .
COPY --from=builder /app/resources/index.bin ./resources/
COPY resources/mcc_risk.json ./resources/
COPY resources/normalization.json ./resources/
EXPOSE 8080
CMD ["./api"]
