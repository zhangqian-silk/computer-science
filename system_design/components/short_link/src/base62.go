package main

import "strings"

const alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

// Encode 把非负整数编码为 62 进制短码。
func Encode(n uint64) string {
	if n == 0 {
		return string(alphabet[0])
	}
	var b strings.Builder
	buf := make([]byte, 0, 11)
	for n > 0 {
		buf = append(buf, alphabet[n%62])
		n /= 62
	}
	for i := len(buf) - 1; i >= 0; i-- { // 逆序输出
		b.WriteByte(buf[i])
	}
	return b.String()
}

// Decode 把 62 进制短码还原为整数。
func Decode(code string) uint64 {
	var n uint64
	for i := 0; i < len(code); i++ {
		n = n*62 + uint64(strings.IndexByte(alphabet, code[i]))
	}
	return n
}
