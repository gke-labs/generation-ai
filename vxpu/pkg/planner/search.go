// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package planner

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/machines"
)

// Layouts and the search over them (#409): the general form of a
// candidate, and a search that explores it with the cost model, for
// vxpu explain --serve to show.
//
// A Layout gives every layer two machines: one for its dense half
// (attention, router, shared expert) and one for its routed experts.
// One-machine, experts-spread and layer-ranges candidates are special
// cases; the search also tries everything in between.

// Layout is, per layer in order, the fleet index of the machine that runs
// its dense half and of the machine that holds its experts.
type Layout struct {
	Dense   []int `json:"dense"`
	Experts []int `json:"experts"`
}

func (l Layout) key() string { return fmt.Sprint(l.Dense, l.Experts) }

func (l Layout) clone() Layout {
	return Layout{Dense: append([]int(nil), l.Dense...), Experts: append([]int(nil), l.Experts...)}
}

// Scored is a layout with its predicted cost.
type Scored struct {
	Layout Layout             `json:"layout"`
	Kind   string             `json:"kind"`
	Ms     float64            `json:"ms_per_token"`
	Parts  map[string]float64 `json:"parts_ms,omitempty"`
	// network crossings a token makes a step
	Hops     int  `json:"hops"`
	Feasible bool `json:"feasible"`
	// why not feasible
	Why string `json:"why,omitempty"`
}

// Scorer prices layouts of an artifact on a fleet, by the cost model's
// rules (Predict's and placeOnGPUs', layer by layer).
type Scorer struct {
	a      Artifact
	fleet  []Node
	ms     []*machines.Machine
	layers []int
	engram map[int]bool
	// a layer's dense bytes a step
	dense    int64
	perToken []int64
	// per layer (in order), its experts' recorded selections, most first,
	// and their total: GPU experts are each machine's hottest
	hot   [][]int64
	total []int64
}

// NewScorer prepares the per-machine rates and the traffic (counts: the
// recorded expert selections, nil for no GPU experts); machines the
// catalog lacks make every layout using them infeasible.
func NewScorer(a Artifact, fleet []Node, counts map[int]map[int]int64) *Scorer {
	s := &Scorer{a: a, fleet: fleet, layers: layersOf(a), engram: map[int]bool{}}
	for _, layer := range s.layers {
		var c []int64
		var t int64
		for _, n := range counts[layer] {
			c, t = append(c, n), t+n
		}
		sort.Slice(c, func(i, j int) bool { return c[i] > c[j] })
		s.hot, s.total = append(s.hot, c), append(s.total, t)
	}
	for _, n := range fleet {
		s.ms = append(s.ms, machines.Match(n.Probe))
	}
	for _, l := range a.Engram {
		s.engram[l] = true
	}
	if len(s.layers) > 0 { // a dense model has no layers to lay out
		s.dense = (a.DecodeFP8Bytes + a.DecodeOtherBytes) / int64(len(s.layers))
	}
	for _, l := range s.layers {
		s.perToken = append(s.perToken, int64(a.TopK)*a.Experts[l].BytesPerExpert)
	}
	return s
}

// Layers is how many layers a layout assigns.
func (s *Scorer) Layers() int { return len(s.layers) }

func (s *Scorer) hasGPU(i int) bool {
	return s.ms[i] != nil && s.ms[i].GPU != nil && s.ms[i].GPUCount > 0
}

// Score prices a layout. Per layer: its dense share on its machine (GPU
// GEMV rate, or the CPU dense rate); its experts on theirs (the CPU
// expert kernel); a crossing per host op; the wire when the experts are
// on another machine than the dense half; a hop when the dense half moves
// to another machine than the previous layer's (and from the last back
// to the first, for the next step); without a GPU (no step graph), the
// launch of each segment. Infeasible: experts beyond a machine's RAM
// (less the dense weights a machine without a GPU holds), dense weights
// beyond its GPU.
func (s *Scorer) Score(l Layout) Scored {
	sc := Scored{Layout: l, Kind: kindOf(l), Parts: map[string]float64{}, Feasible: true}
	gpu := s.gpuExperts(l)
	ram := make([]int64, len(s.fleet))
	denseOn := make([]int64, len(s.fleet))
	for i, layer := range s.layers {
		d, e := l.Dense[i], l.Experts[i]
		if s.ms[d] == nil || s.ms[e] == nil {
			sc.Feasible, sc.Why = false, "a machine not in the catalog"
			return sc
		}
		ram[e] += int64(s.a.Experts[layer].Experts) * s.a.Experts[layer].BytesPerExpert
		denseOn[d] += s.a.DenseBytes / int64(len(s.layers))
		if s.hasGPU(d) {
			sc.Parts["dense"] += ms(s.dense, gemvBandwidth(s.ms[d]))
		} else {
			bw, _ := cpuDenseBandwidth(s.ms[d])
			sc.Parts["dense"] += ms(s.dense, bw)
		}
		host := 1.0
		if s.engram[layer] {
			host++
			sc.Parts["n-gram"] += engramMs
		}
		sc.Parts["crossings"] += host * crossingMs
		if !s.hasGPU(d) {
			sc.Parts["segments"] += host * segmentMs
		}
		t := ms(s.perToken[i], cpuKernelBandwidth(s.ms[e]))
		if g, ok := gpu[i]; ok && e == d { // its hottest experts on the machine's GPUs (Predict's rule)
			m := s.ms[d]
			t = max(ms(int64((1-g.share)*float64(s.perToken[i])), cpuKernelBandwidth(m)),
				ms(int64(g.slowest*float64(s.perToken[i])), gemvEfficiency*m.GPU.MemoryBandwidthBytesPerSec)) + gpuExpertsMs
			if g.peer {
				t += peerGPUMs
			}
		}
		if e != d {
			sc.Parts["experts remote"] += t
			sc.Parts["wire"] += remoteMs
			sc.Hops += 2
		} else {
			sc.Parts["experts local"] += t
		}
		prev := l.Dense[(i+len(s.layers)-1)%len(s.layers)]
		if prev != d {
			sc.Parts["hops"] += remoteMs
			sc.Hops++
			if !s.a.CanStart(layer) && sc.Feasible {
				sc.Feasible, sc.Why = false, fmt.Sprintf("a range can't start at layer %d: %s", layer, s.a.Uncuttable[layer])
			}
		}
	}
	for i := range s.fleet {
		budget := ramBudget(s.fleet[i].Probe.RamBytes)
		if !s.hasGPU(i) {
			budget -= denseOn[i]
		} else if need := int64(denseFP8Factor * float64(denseOn[i])); need > s.fleet[i].Probe.GpuMemoryBytes-gpuHeadroomBytes {
			sc.Feasible, sc.Why = false, fmt.Sprintf("%s: %.1f GB of dense weights on its GPU", s.fleet[i].Name, gb(need))
		}
		if ram[i] > budget {
			sc.Feasible, sc.Why = false, fmt.Sprintf("%s: %.0f GB of experts in %.0f GB of RAM", s.fleet[i].Name, gb(ram[i]), gb(max(budget, 0)))
		}
	}
	for _, v := range sc.Parts {
		sc.Ms += v
	}
	return sc
}

type gpuHeld struct {
	// of the layer's selections its GPU experts serve
	share float64
	// the largest share on one GPU: the GPUs run their shares in parallel
	slowest float64
	// some on a GPU other than the dense half's
	peer bool
}

// gpuExperts places each GPU machine's hottest experts, of the layers
// whose dense half and experts it both runs (node servers hold no GPU
// experts), on its GPUs as placeOnGPUs does: GPU0 first, less the dense
// weights it holds, then the others. Returns, by layer position, the
// share each layer's GPU experts serve.
func (s *Scorer) gpuExperts(l Layout) map[int]gpuHeld {
	out := map[int]gpuHeld{}
	if len(s.hot) == 0 {
		return out
	}
	for m := range s.fleet {
		if !s.hasGPU(m) {
			continue
		}
		var mine []int
		var dense int64
		for i := range s.layers {
			if l.Dense[i] == m {
				dense += s.a.DenseBytes / int64(len(s.layers))
				if l.Experts[i] == m && s.total[i] > 0 {
					mine = append(mine, i)
				}
			}
		}
		if len(mine) == 0 {
			continue
		}
		var perExpert int64 // the largest of its layers', as placeOnGPUs sizes the slots
		for _, i := range mine {
			perExpert = max(perExpert, s.a.Experts[s.layers[i]].BytesPerExpert)
		}
		probe := s.fleet[m].Probe
		budgets := make([]int64, probe.GpuCount)
		for g := range budgets {
			budgets[g] = probe.GpuMemoryBytes - gpuHeadroomBytes
		}
		factor := denseFactor
		if budgets[0] < int64(denseFactor*float64(dense)) {
			factor = denseFP8Factor
		}
		budgets[0] = max(0, budgets[0]-int64(factor*float64(dense)))
		// the hottest experts across the machine's layers, in order (a k-way merge), filling GPU0 then the others
		h := &hotHeap{s: s}
		for _, i := range mine {
			heap.Push(h, hotItem{i, 0})
		}
		served := map[int][]int64{} // layer -> per GPU, the selections its experts there serve
		for g, b := range budgets {
			for n := b / perExpert; n > 0 && h.Len() > 0; n-- {
				it := heap.Pop(h).(hotItem)
				if served[it.layer] == nil {
					served[it.layer] = make([]int64, len(budgets))
				}
				served[it.layer][g] += s.hot[it.layer][it.k]
				if it.k+1 < len(s.hot[it.layer]) {
					heap.Push(h, hotItem{it.layer, it.k + 1})
				}
			}
		}
		for i, per := range served {
			var g gpuHeld
			for gi, n := range per {
				share := float64(n) / float64(s.total[i])
				g.share += share
				g.slowest = max(g.slowest, share)
				g.peer = g.peer || (gi > 0 && n > 0)
			}
			out[i] = g
		}
	}
	return out
}

type hotItem struct{ layer, k int }

// hotHeap orders each layer's next hottest expert, the most selected first.
type hotHeap struct {
	s     *Scorer
	items []hotItem
}

func (h *hotHeap) Len() int { return len(h.items) }
func (h *hotHeap) Less(i, j int) bool {
	return h.s.hot[h.items[i].layer][h.items[i].k] > h.s.hot[h.items[j].layer][h.items[j].k]
}
func (h *hotHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }
func (h *hotHeap) Push(x any)    { h.items = append(h.items, x.(hotItem)) }
func (h *hotHeap) Pop() any {
	it := h.items[len(h.items)-1]
	h.items = h.items[:len(h.items)-1]
	return it
}

// kindOf names a layout's shape.
func kindOf(l Layout) string {
	sameDense, local, ranges := true, true, 0
	for i := range l.Dense {
		sameDense = sameDense && l.Dense[i] == l.Dense[0]
		local = local && l.Experts[i] == l.Dense[i]
		if l.Dense[i] != l.Dense[(i+len(l.Dense)-1)%len(l.Dense)] { // cyclic: the step wraps to the first layer
			ranges++
		}
	}
	switch {
	case sameDense && local:
		return "one-machine"
	case sameDense:
		return "experts-spread"
	case local:
		return fmt.Sprintf("layer-ranges (%d)", ranges)
	}
	return "mixed"
}

// Search explores layouts: random ones, and changes to the current one
// accepted as in simulated annealing, keeping the best N. Step proposes,
// scores and keeps one.
type Search struct {
	s       *Scorer
	rng     *rand.Rand
	cur     Scored
	temp    float64
	Top     []Scored
	keep    int
	seen    map[string]bool
	Scanned int
	Valid   int
	steps   int
}

// NewSearch starts a search keeping the best keep layouts.
func NewSearch(s *Scorer, keep int, seed int64) *Search {
	sr := &Search{s: s, rng: rand.New(rand.NewSource(seed)), keep: keep, seen: map[string]bool{}, temp: 1}
	sr.cur = sr.s.Score(sr.randomRanges())
	return sr
}

// StartFrom moves the search to l (a good layout to perturb), keeping it
// if it is among the best.
func (sr *Search) StartFrom(l Layout) {
	sc := sr.s.Score(l)
	if sc.Feasible {
		sr.cur = sc
		sr.offer(sc)
	}
}

// LayoutOf is a candidate's placements as a layout over the fleet.
func (s *Scorer) LayoutOf(c Candidate) Layout {
	index := map[string]int{}
	for i, n := range s.fleet {
		index[n.Name] = i
	}
	pos := map[int]int{}
	for i, layer := range s.layers {
		pos[layer] = i
	}
	l := Layout{Dense: make([]int, len(s.layers)), Experts: make([]int, len(s.layers))}
	engine := -1
	for _, p := range c.Placements {
		if p.Role == "engine" {
			engine = index[p.Node]
		}
	}
	for _, p := range c.Placements {
		for _, layer := range p.Layers {
			l.Experts[pos[layer]] = index[p.Node]
			l.Dense[pos[layer]] = index[p.Node]
			if engine >= 0 {
				l.Dense[pos[layer]] = engine
			}
		}
	}
	return l
}

// Step proposes a layout, scores it, keeps it if it is among the best,
// and moves the search to it as annealing allows. Returns it.
func (sr *Search) Step() Scored {
	sr.steps++
	var l Layout
	switch r := sr.rng.Float64(); {
	case r < 0.08:
		l = sr.randomRanges()
	case r < 0.14:
		l = sr.randomSpread()
	case r < 0.18:
		l = sr.randomAny()
	default:
		l = sr.perturb(sr.cur.Layout)
	}
	sc := sr.s.Score(l)
	sr.Scanned++
	if sc.Feasible {
		sr.Valid++
		sr.offer(sc)
		// annealing on the log of the cost: a layout twice as slow is as unlikely at any scale
		if !sr.cur.Feasible || sc.Ms < sr.cur.Ms || sr.rng.Float64() < math.Exp(-math.Log(sc.Ms/sr.cur.Ms)/sr.temp) {
			sr.cur = sc
		}
	}
	sr.temp = max(0.02, sr.temp*0.995)
	if sr.steps%600 == 0 { // start over from somewhere new now and then: the display keeps searching
		sr.temp, sr.cur = 1, sr.s.Score(sr.randomRanges())
	}
	return sc
}

// canonical is a layout's key with machines of the same type
// interchangeable (renumbered by first use), so the best kept are
// different layouts, not one layout on two identical machines.
func (sr *Search) canonical(l Layout) string {
	label := map[int]int{}
	byType := map[string]int{}
	relabel := func(i int) int {
		if v, ok := label[i]; ok {
			return v
		}
		t := machineName(sr.s.fleet[i].Probe)
		v := byType[t]*1000 + typeIndex(sr.s.fleet, t)
		byType[t]++
		label[i] = v
		return v
	}
	out := Layout{Dense: make([]int, len(l.Dense)), Experts: make([]int, len(l.Experts))}
	for i := range l.Dense {
		out.Dense[i] = relabel(l.Dense[i])
		out.Experts[i] = relabel(l.Experts[i])
	}
	return out.key()
}

func typeIndex(fleet []Node, t string) int {
	for i, n := range fleet {
		if machineName(n.Probe) == t {
			return i
		}
	}
	return -1
}

func (sr *Search) offer(sc Scored) {
	k := sr.canonical(sc.Layout)
	if sr.seen[k] {
		return
	}
	if len(sr.Top) == sr.keep && sc.Ms >= sr.Top[len(sr.Top)-1].Ms {
		return
	}
	sr.seen[k] = true
	sr.Top = append(sr.Top, sc)
	sort.SliceStable(sr.Top, func(i, j int) bool { return sr.Top[i].Ms < sr.Top[j].Ms })
	if len(sr.Top) > sr.keep {
		delete(sr.seen, sr.canonical(sr.Top[sr.keep].Layout))
		sr.Top = sr.Top[:sr.keep]
	}
}

func (sr *Search) n() int { return len(sr.s.fleet) }

// randomRanges: a random number of contiguous ranges of whole layers on
// random machines.
func (sr *Search) randomRanges() Layout {
	L := sr.s.Layers()
	k := 1 + sr.rng.Intn(max(1, min(sr.n(), 4, L))) // no more ranges than layers
	cuts := sr.rng.Perm(L - 1)[:k-1]
	sort.Ints(cuts)
	l := Layout{Dense: make([]int, L), Experts: make([]int, L)}
	machine, c := sr.rng.Intn(sr.n()), 0
	for i := range L {
		if c < len(cuts) && i == cuts[c]+1 {
			machine, c = sr.rng.Intn(sr.n()), c+1
		}
		l.Dense[i], l.Experts[i] = machine, machine
	}
	return l
}

// randomSpread: one machine runs every dense half; contiguous blocks of
// experts on random machines.
func (sr *Search) randomSpread() Layout {
	l := sr.randomRanges()
	engine := sr.rng.Intn(sr.n())
	for i := range l.Dense {
		l.Dense[i] = engine
	}
	return l
}

// randomAny: every layer's two halves anywhere (mostly hopeless: noise).
func (sr *Search) randomAny() Layout {
	L := sr.s.Layers()
	l := Layout{Dense: make([]int, L), Experts: make([]int, L)}
	for i := range L {
		l.Dense[i], l.Experts[i] = sr.rng.Intn(sr.n()), sr.rng.Intn(sr.n())
	}
	return l
}

// perturb changes a layout a little: a boundary moves, a block of layers
// moves to another machine (both halves, or only the experts or only the
// dense halves), or two machines trade places.
func (sr *Search) perturb(base Layout) Layout {
	l := base.clone()
	L := len(l.Dense)
	i := sr.rng.Intn(L)
	j := min(L, i+1+sr.rng.Intn(max(1, L/4)))
	to := sr.rng.Intn(sr.n())
	switch sr.rng.Intn(5) {
	case 0: // move a range boundary: the layer takes its neighbour's machines
		if i > 0 {
			l.Dense[i], l.Experts[i] = l.Dense[i-1], l.Experts[i-1]
		}
	case 1: // a block, both halves
		for k := i; k < j; k++ {
			l.Dense[k], l.Experts[k] = to, to
		}
	case 2: // a block's experts
		for k := i; k < j; k++ {
			l.Experts[k] = to
		}
	case 3: // a block's dense halves
		for k := i; k < j; k++ {
			l.Dense[k] = to
		}
	case 4: // two machines trade places everywhere
		a, b := sr.rng.Intn(sr.n()), to
		for k := range L {
			l.Dense[k] = swap(l.Dense[k], a, b)
			l.Experts[k] = swap(l.Experts[k], a, b)
		}
	}
	return l
}

func swap(x, a, b int) int {
	switch x {
	case a:
		return b
	case b:
		return a
	}
	return x
}

// Describe is a layout in words: which machine runs which layers.
func (s *Scorer) Describe(l Layout) string {
	var parts []string
	for i := 0; i < len(l.Dense); {
		j := i
		for j < len(l.Dense) && l.Dense[j] == l.Dense[i] && l.Experts[j] == l.Experts[i] {
			j++
		}
		d, e := s.fleet[l.Dense[i]].Name, s.fleet[l.Experts[i]].Name
		where := d
		if d != e {
			where = d + " (experts " + e + ")"
		}
		parts = append(parts, fmt.Sprintf("%s: %s", span(s.layers[i:j]), where))
		i = j
	}
	return strings.Join(parts, ", ")
}
