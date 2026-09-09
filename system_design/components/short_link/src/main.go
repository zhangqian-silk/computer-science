package main

import "fmt"

func main() {
	// base62 编解码往返
	for _, id := range []uint64{1, 10000, 1000000, 100000000, 1000000000} {
		code := Encode(id)
		fmt.Printf("ID %-11d -> %-6s (%d 位) -> Decode %d\n", id, code, len(code), Decode(code))
	}

	// 短链服务：从一个较大的起始 ID 开始发号，短码更接近真实长度
	fmt.Println("\n短链服务：")
	s := NewShortener(1000000)
	urls := []string{
		"https://example.com/products/12345?utm=spring",
		"https://example.com/articles/distributed-id",
		"https://example.com/products/12345?utm=spring", // 重复长链，复用短码
	}
	for _, u := range urls {
		fmt.Printf("  sho.rt/%s  <-  %s\n", s.Shorten(u), u)
	}
	if long, ok := s.Resolve(s.Shorten(urls[0])); ok {
		fmt.Printf("\n解析短码得到长链：%s\n", long)
	}

	// 防枚举：连续 ID 扰动后短码不再连续，但可无损还原
	fmt.Println("\n防枚举扰动（连续 ID -> 无序短码，且可还原）：")
	ok := true
	for _, id := range []int64{1000000, 1000001, 1000002, 1000003} {
		s := Obfuscate(id)
		code := Encode(uint64(s))
		back := Deobfuscate(s)
		if back != id {
			ok = false
		}
		fmt.Printf("  ID %d -> 扰动 %d -> sho.rt/%s -> 还原 %d\n", id, s, code, back)
	}
	fmt.Printf("往返一致：%v（连续 ID 的短码已不连续，无法顺序枚举）\n", ok)
}
