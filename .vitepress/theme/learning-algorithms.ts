export type Merge = { pair: [string, string]; count: number }
export function mergePair(tokens: string[], pair: readonly string[]) {
	const output: string[] = []
	for (let i = 0; i < tokens.length; i++) {
		if (tokens[i] === pair[0] && tokens[i + 1] === pair[1]) { output.push(tokens[i] + tokens[i + 1]); i++ }
		else output.push(tokens[i])
	}
	return output
}
export function trainBPE(corpus: { word: string; count: number }[], limit: number) {
	let words = corpus.map(item => ({ ...item, tokens: [...item.word, "</w>"] }))
	const rules: Merge[] = []
	const stages = []
	for (let step = 0; step <= limit; step++) {
		const pairs = new Map<string, Merge>()
		for (const item of words) {
			for (let i = 0; i < item.tokens.length - 1; i++) {
				const pair: [string, string] = [item.tokens[i], item.tokens[i + 1]]
				const key = JSON.stringify(pair)
				pairs.set(key, { pair, count: (pairs.get(key)?.count ?? 0) + item.count })
			}
		}
		const candidates = [...pairs.values()].sort((a, b) => b.count - a.count || JSON.stringify(a.pair).localeCompare(JSON.stringify(b.pair), "en"))
		stages.push({ words, candidates, rules: [...rules] })
		if (step === limit || candidates.length === 0) break
		rules.push(candidates[0])
		words = words.map(item => ({ ...item, tokens: mergePair(item.tokens, candidates[0].pair) }))
	}
	return stages
}
export function encodeBPE(word: string, rules: Merge[]) {
	return rules.reduce((tokens, rule) => mergePair(tokens, rule.pair), [...word, "</w>"])
}
export type Beam = { tokens: string[]; probability: number }
export function toyNext(prefix: string[]): [string, number][] {
	if (prefix.length === 0) return [["I", .6], ["We", .4]]
	if (prefix.length === 1) return prefix[0] === "We" ? [["love", .95], ["like", .05]] : [["like", .6], ["love", .4]]
	if (prefix.length === 2) return prefix[0] === "We" && prefix[1] === "love" ? [["AI", .9], ["math", .1]] : [["math", .55], ["AI", .45]]
	return [["<EOS>", 1]]
}
export function beamSteps(width: number, steps: number) {
	let beams: Beam[] = [{ tokens: [], probability: 1 }]
	const history = [beams]
	for (let step = 0; step < steps; step++) {
		beams = beams.flatMap(beam => beam.tokens.at(-1) === "<EOS>" ? [beam] : toyNext(beam.tokens).map(([token, p]) => ({
			tokens: [...beam.tokens, token], probability: beam.probability * p
		}))).sort((a, b) => b.probability - a.probability).slice(0, width)
		history.push(beams)
	}
	return history
}
export function scheduleToy(mode: "static" | "continuous", shortLength: number) {
	const jobs = [{ id: "A", arrival: 0, work: 4 }, { id: "B", arrival: 0, work: shortLength }, { id: "C", arrival: 0, work: 6 }, { id: "D", arrival: 1, work: 3 }]
	const remaining = jobs.map(job => job.work)
	const started = jobs.map(() => -1)
	const rows: string[][] = jobs.map(() => [])
	let active: number[] = []
	for (let round = 0; round < 16 && remaining.some(value => value > 0); round++) {
		if (mode === "continuous" || active.length === 0) {
			for (let i = 0; i < jobs.length && active.length < 3; i++) {
				if (started[i] === -1 && jobs[i].arrival <= round) { active.push(i); started[i] = round }
			}
		}
		for (let i = 0; i < jobs.length; i++) {
			rows[i].push(active.includes(i) ? (started[i] === round ? "P" : "D") : started[i] === -1 && jobs[i].arrival <= round ? "等" : "·")
			if (active.includes(i)) remaining[i]--
		}
		active = active.filter(i => remaining[i] > 0)
	}
	return { jobs, rows, started }
}
