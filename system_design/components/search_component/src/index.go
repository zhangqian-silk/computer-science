package main

import (
	"math"
	"sort"
	"strings"
)

// InvertedIndex 是教学级倒排索引：term -> 文档词频，支持 TF-IDF 排序检索。
type InvertedIndex struct {
	postings map[string]map[int]int // term -> (docID -> 词频)
	docLen   map[int]int
	docs     map[int]string
}

func NewInvertedIndex() *InvertedIndex {
	return &InvertedIndex{
		postings: make(map[string]map[int]int),
		docLen:   make(map[int]int),
		docs:     make(map[int]string),
	}
}

// tokenize 是简化分词：按空格切分并小写化。
func tokenize(text string) []string {
	return strings.Fields(strings.ToLower(text))
}

// Add 索引一篇文档。
func (ix *InvertedIndex) Add(id int, text string) {
	ix.docs[id] = text
	terms := tokenize(text)
	ix.docLen[id] = len(terms)
	for _, t := range terms {
		if ix.postings[t] == nil {
			ix.postings[t] = make(map[int]int)
		}
		ix.postings[t][id]++
	}
}

func (ix *InvertedIndex) idf(term string) float64 {
	df := len(ix.postings[term])
	if df == 0 {
		return 0
	}
	return math.Log(float64(len(ix.docs))/float64(df)) + 1
}

// bm25IDF 是 BM25 的 idf 变体：ln((N - df + 0.5)/(df + 0.5) + 1)。
func (ix *InvertedIndex) bm25IDF(term string) float64 {
	df := float64(len(ix.postings[term]))
	if df == 0 {
		return 0
	}
	n := float64(len(ix.docs))
	return math.Log((n-df+0.5)/(df+0.5) + 1)
}

func (ix *InvertedIndex) avgDocLen() float64 {
	if len(ix.docLen) == 0 {
		return 0
	}
	var total int
	for _, l := range ix.docLen {
		total += l
	}
	return float64(total) / float64(len(ix.docLen))
}

// Result 是一条检索结果。
type Result struct {
	DocID int
	Score float64
}

// Search 对查询分词后用 TF-IDF 累加打分，按分数降序返回。
// requireAll=true 时要求文档包含全部查询词（AND），否则任一命中即可（OR）。
func (ix *InvertedIndex) Search(query string, requireAll bool) []Result {
	terms := tokenize(query)
	scores := make(map[int]float64)
	matched := make(map[int]int)
	for _, t := range terms {
		idf := ix.idf(t)
		for docID, tf := range ix.postings[t] {
			scores[docID] += float64(tf) * idf
			matched[docID]++
		}
	}
	var out []Result
	for docID, s := range scores {
		if requireAll && matched[docID] < len(terms) {
			continue
		}
		out = append(out, Result{DocID: docID, Score: s})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].DocID < out[j].DocID
	})
	return out
}

// SearchBM25 用 BM25 打分：对词频饱和 + 文档长度归一化，长文档不再因词多而虚高。
func (ix *InvertedIndex) SearchBM25(query string) []Result {
	const k1, b = 1.2, 0.75
	avg := ix.avgDocLen()
	terms := tokenize(query)
	scores := make(map[int]float64)
	for _, t := range terms {
		idf := ix.bm25IDF(t)
		for docID, tf := range ix.postings[t] {
			f := float64(tf)
			dl := float64(ix.docLen[docID])
			scores[docID] += idf * (f * (k1 + 1)) / (f + k1*(1-b+b*dl/avg))
		}
	}
	var out []Result
	for docID, s := range scores {
		out = append(out, Result{DocID: docID, Score: s})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].DocID < out[j].DocID
	})
	return out
}
