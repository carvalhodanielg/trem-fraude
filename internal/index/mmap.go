package index

import (
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

// mmapReadOnly mapeia o arquivo em path para leitura compartilhada (MAP_SHARED).
// Após a chamada o file descriptor pode ser fechado — o mapeamento sobrevive.
// O slice retornado é válido enquanto o Index estiver vivo.
// Em Docker com overlayfs (mesma imagem), dois containers mapeando o mesmo arquivo
// da layer read-only do overlay compartilham as mesmas physical pages no page cache
// do kernel — a memória física do índice (~122 MB) é paga uma única vez.
func mmapReadOnly(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("mmap open %s: %w", path, err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("mmap stat %s: %w", path, err)
	}
	size := int(fi.Size())
	if size == 0 {
		return nil, fmt.Errorf("mmap: arquivo vazio: %s", path)
	}

	// MAP_SHARED: alterações (não ocorrem aqui) seriam propagadas ao arquivo.
	// PROT_READ:  somente leitura — qualquer write causa SIGSEGV (segurança extra).
	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return data, nil
}
