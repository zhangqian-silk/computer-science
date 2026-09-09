package main

// 防枚举：对自增 ID 做「乘法逆元」可逆扰动，使短码看起来无序但仍双射无冲突。
// 取模数 M = 62^7、常数 K 与 M 互质（M = 2^7 * 31^7，故 K 取奇数且非 31 的倍数）。
const (
	obfM int64 = 3521614606208 // 62^7
	obfK int64 = 1580030173    // 与 M 互质
)

var obfKInv = modInverse(obfK, obfM)

// Obfuscate 把连续 id 打散：s = id * K mod M。
func Obfuscate(id int64) int64 { return mulmod(id, obfK, obfM) }

// Deobfuscate 还原：id = s * K^{-1} mod M。
func Deobfuscate(s int64) int64 { return mulmod(s, obfKInv, obfM) }

// mulmod 计算 (a*b) mod m，用二进制乘法避免 int64 溢出。
func mulmod(a, b, m int64) int64 {
	a %= m
	var res int64
	for b > 0 {
		if b&1 == 1 {
			res = (res + a) % m
		}
		a = (a * 2) % m
		b >>= 1
	}
	return res
}

// modInverse 用扩展欧几里得求 a 在模 m 下的乘法逆元。
func modInverse(a, m int64) int64 {
	g, x, _ := extGCD(a%m, m)
	if g != 1 {
		panic("K 与 M 不互质，无法求逆元")
	}
	return (x%m + m) % m
}

func extGCD(a, b int64) (g, x, y int64) {
	if b == 0 {
		return a, 1, 0
	}
	g, x1, y1 := extGCD(b, a%b)
	return g, y1, x1 - (a/b)*y1
}
