package main

import "fmt"

func main() {
	n := NewNotifier(2) // 单用户单渠道每天最多 2 条

	type req struct{ event, user, channel string }
	reqs := []req{
		{"e1", "u1", "sms", "验证码"},
		{"e1", "u1", "sms", "验证码"}, // 重复事件 -> 去重
		{"e2", "u1", "sms", "营销1"},
		{"e3", "u1", "sms", "营销2"}, // 达到频控上限
		{"e4", "u1", "sms", "营销3"}, // 被频控拦截
		{"e4", "u1", "push", "营销3"}, // 换渠道 -> 独立计数, 送达
	}

	fmt.Printf("通知中心 %s\n", n)
	for _, r := range reqs {
		res := n.Send(r.event, r.user, r.channel, r.channel)
		fmt.Printf("  %s/%s/%-4s -> %s\n", r.event, r.user, r.channel, res)
	}
}
