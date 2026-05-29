FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o api ./cmd/api
RUN go build -o preprocess ./cmd/preprocess

FROM alpine:3.19
WORKDIR /app
COPY --from=builder /app/api .
COPY resources/index.bin ./resources/
COPY resources/mcc_risk.json ./resources/
COPY resources/normalization.json ./resources/
EXPOSE 8080
CMD ["./api"]