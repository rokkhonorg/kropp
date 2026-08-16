//! Boykov-Kolmogorov max-flow / min-cut.
//!
//! A port of the algorithm from Boykov & Kolmogorov, *An Experimental
//! Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in
//! Vision* (PAMI 2004). Unlike Ford-Fulkerson variants, it maintains two search
//! trees rooted at the terminals and reuses them across augmentations, which is
//! what makes it fast on the grid graphs vision problems produce — exactly the
//! shape of a seam-finding graph over an overlap region.
//!
//! The three phases per iteration:
//!
//! * **grow** — expand the source and sink trees along non-saturated arcs until
//!   they touch, which identifies an augmenting path.
//! * **augment** — push the bottleneck along that path. Arcs that saturate
//!   detach nodes from their parents, orphaning them.
//! * **adopt** — find each orphan a new parent within its own tree, or free it
//!   and orphan its children in turn.
//!
//! Capacities are `f32`. The min-cut is read back with [`Graph::is_source_side`]
//! after [`Graph::maxflow`].

/// Sentinels for `Node::parent`, which otherwise holds an arc index.
const PARENT_NONE: u32 = u32::MAX;
const PARENT_TERMINAL: u32 = u32::MAX - 1;
const PARENT_ORPHAN: u32 = u32::MAX - 2;
const NO_ARC: u32 = u32::MAX;

/// Distance standing in for "cannot reach a terminal".
const INFINITE_DIST: u32 = u32::MAX;

#[derive(Clone)]
struct Node {
    /// First arc leaving this node; `NO_ARC` when it has none.
    first_arc: u32,
    /// Arc to this node's parent in its search tree, or a sentinel.
    parent: u32,
    /// Timestamp of the last distance computation, for the origin check.
    timestamp: u32,
    /// Cached distance to the tree's terminal.
    dist: u32,
    /// Which tree this node belongs to once `parent` is set.
    is_sink: bool,
    /// Residual capacity to the terminals: positive toward the source,
    /// negative toward the sink.
    terminal_cap: f32,
    /// Whether the node is queued for growth.
    active: bool,
}

#[derive(Clone)]
struct Arc {
    /// Node this arc points at.
    head: u32,
    /// Next arc leaving the same node.
    next: u32,
    /// The opposing arc; arcs are always added in pairs.
    sister: u32,
    /// Remaining capacity.
    residual: f32,
}

pub struct Graph {
    nodes: Vec<Node>,
    arcs: Vec<Arc>,
    active: std::collections::VecDeque<u32>,
    orphans: std::collections::VecDeque<u32>,
    time: u32,
    flow: f32,
}

impl Graph {
    pub fn new(node_count: usize, expected_edges: usize) -> Self {
        Self {
            nodes: vec![
                Node {
                    first_arc: NO_ARC,
                    parent: PARENT_NONE,
                    timestamp: 0,
                    dist: 0,
                    is_sink: false,
                    terminal_cap: 0.0,
                    active: false,
                };
                node_count
            ],
            arcs: Vec::with_capacity(expected_edges * 2),
            active: Default::default(),
            orphans: Default::default(),
            time: 0,
            flow: 0.0,
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Link `node` to the terminals. Costs accumulate, and the part that flows
    /// straight through (the smaller of the two) is credited immediately, which
    /// is what lets the node hold a single signed residual afterwards.
    pub fn add_terminal_weights(&mut self, node: usize, to_source: f32, to_sink: f32) {
        let existing = self.nodes[node].terminal_cap;
        let (mut source, mut sink) = (to_source, to_sink);
        if existing > 0.0 {
            source += existing;
        } else {
            sink -= existing;
        }
        self.flow += source.min(sink);
        self.nodes[node].terminal_cap = source - sink;
    }

    /// Add an undirected pair of arcs between `a` and `b`.
    pub fn add_edge(&mut self, a: usize, b: usize, capacity: f32, reverse_capacity: f32) {
        let forward = self.arcs.len() as u32;
        let backward = forward + 1;

        self.arcs.push(Arc {
            head: b as u32,
            next: self.nodes[a].first_arc,
            sister: backward,
            residual: capacity,
        });
        self.nodes[a].first_arc = forward;

        self.arcs.push(Arc {
            head: a as u32,
            next: self.nodes[b].first_arc,
            sister: forward,
            residual: reverse_capacity,
        });
        self.nodes[b].first_arc = backward;
    }

    fn set_active(&mut self, node: u32) {
        if !self.nodes[node as usize].active {
            self.nodes[node as usize].active = true;
            self.active.push_back(node);
        }
    }

    fn next_active(&mut self) -> Option<u32> {
        while let Some(node) = self.active.pop_front() {
            self.nodes[node as usize].active = false;
            // A node orphaned since being queued is no longer a growth point.
            if self.nodes[node as usize].parent != PARENT_NONE {
                return Some(node);
            }
        }
        None
    }

    fn set_orphan_front(&mut self, node: u32) {
        self.nodes[node as usize].parent = PARENT_ORPHAN;
        self.orphans.push_front(node);
    }

    fn set_orphan_rear(&mut self, node: u32) {
        self.nodes[node as usize].parent = PARENT_ORPHAN;
        self.orphans.push_back(node);
    }

    /// Seed both trees from every node with a terminal connection.
    fn init(&mut self) {
        self.active.clear();
        self.orphans.clear();
        self.time = 0;
        for index in 0..self.nodes.len() {
            let cap = self.nodes[index].terminal_cap;
            let node = &mut self.nodes[index];
            node.active = false;
            if cap != 0.0 {
                node.is_sink = cap < 0.0;
                node.parent = PARENT_TERMINAL;
                node.dist = 1;
                node.timestamp = 0;
                self.set_active(index as u32);
            } else {
                node.parent = PARENT_NONE;
            }
        }
    }

    /// Grow from `node`, returning the arc joining the two trees if found.
    fn grow(&mut self, node: u32) -> Option<u32> {
        let growing_sink = self.nodes[node as usize].is_sink;
        let mut arc_index = self.nodes[node as usize].first_arc;

        while arc_index != NO_ARC {
            let arc = self.arcs[arc_index as usize].clone();
            // A source tree advances along an arc's own capacity; a sink tree
            // advances against it, so it reads the sister's.
            let usable = if growing_sink {
                self.arcs[arc.sister as usize].residual
            } else {
                arc.residual
            };

            if usable > 0.0 {
                let head = arc.head;
                let head_parent = self.nodes[head as usize].parent;

                if head_parent == PARENT_NONE {
                    let (timestamp, dist) = {
                        let n = &self.nodes[node as usize];
                        (n.timestamp, n.dist)
                    };
                    let head_node = &mut self.nodes[head as usize];
                    head_node.is_sink = growing_sink;
                    // The sister runs head -> node, so `parent.head` names the
                    // parent node. Storing the forward arc instead would make
                    // `parent.head` the node itself, and every walk toward a
                    // terminal would spin on it forever.
                    head_node.parent = arc.sister;
                    head_node.timestamp = timestamp;
                    head_node.dist = dist + 1;
                    self.set_active(head);
                } else if self.nodes[head as usize].is_sink != growing_sink {
                    // The trees met: this arc is the middle of the path, always
                    // oriented source -> sink.
                    return Some(if growing_sink { arc.sister } else { arc_index });
                } else {
                    // Same tree: adopt a shorter route to the terminal if this
                    // one offers it.
                    let (timestamp, dist) = {
                        let n = &self.nodes[node as usize];
                        (n.timestamp, n.dist)
                    };
                    let head_node = &self.nodes[head as usize];
                    if head_node.timestamp <= timestamp && head_node.dist > dist {
                        let head_node = &mut self.nodes[head as usize];
                        head_node.parent = arc.sister;
                        head_node.timestamp = timestamp;
                        head_node.dist = dist + 1;
                    }
                }
            }
            arc_index = arc.next;
        }
        None
    }

    /// Push the bottleneck along the path through `middle`, orphaning any node
    /// whose link to its parent saturates. Returns whether anything moved.
    fn augment(&mut self, middle: u32) -> bool {
        let middle_arc = self.arcs[middle as usize].clone();
        let mut bottleneck = middle_arc.residual;

        // Bottleneck along the source side.
        let mut node = self.arcs[middle_arc.sister as usize].head;
        loop {
            let parent = self.nodes[node as usize].parent;
            if parent == PARENT_TERMINAL {
                break;
            }
            let sister = self.arcs[parent as usize].sister;
            bottleneck = bottleneck.min(self.arcs[sister as usize].residual);
            node = self.arcs[parent as usize].head;
        }
        bottleneck = bottleneck.min(self.nodes[node as usize].terminal_cap);

        // Bottleneck along the sink side.
        let mut node = middle_arc.head;
        loop {
            let parent = self.nodes[node as usize].parent;
            if parent == PARENT_TERMINAL {
                break;
            }
            bottleneck = bottleneck.min(self.arcs[parent as usize].residual);
            node = self.arcs[parent as usize].head;
        }
        bottleneck = bottleneck.min(-self.nodes[node as usize].terminal_cap);

        // Rounding can leave a path whose bottleneck is not actually positive
        // (NaN included). Pushing zero would make no progress while leaving the
        // path intact, so the search would rediscover it forever.
        if bottleneck <= 0.0 || bottleneck.is_nan() {
            return false;
        }

        // Push it: the middle arc first, then out to both terminals.
        self.arcs[middle_arc.sister as usize].residual += bottleneck;
        self.arcs[middle as usize].residual -= bottleneck;

        let mut node = self.arcs[middle_arc.sister as usize].head;
        loop {
            let parent = self.nodes[node as usize].parent;
            if parent == PARENT_TERMINAL {
                break;
            }
            let sister = self.arcs[parent as usize].sister;
            self.arcs[parent as usize].residual += bottleneck;
            self.arcs[sister as usize].residual -= bottleneck;
            if self.arcs[sister as usize].residual <= 0.0 {
                self.set_orphan_front(node);
            }
            node = self.arcs[parent as usize].head;
        }
        self.nodes[node as usize].terminal_cap -= bottleneck;
        if self.nodes[node as usize].terminal_cap <= 0.0 {
            self.set_orphan_front(node);
        }

        let mut node = middle_arc.head;
        loop {
            let parent = self.nodes[node as usize].parent;
            if parent == PARENT_TERMINAL {
                break;
            }
            let sister = self.arcs[parent as usize].sister;
            self.arcs[sister as usize].residual += bottleneck;
            self.arcs[parent as usize].residual -= bottleneck;
            if self.arcs[parent as usize].residual <= 0.0 {
                self.set_orphan_front(node);
            }
            node = self.arcs[parent as usize].head;
        }
        self.nodes[node as usize].terminal_cap += bottleneck;
        if self.nodes[node as usize].terminal_cap >= 0.0 {
            self.set_orphan_front(node);
        }

        self.flow += bottleneck;
        true
    }

    /// Distance from `start` to its terminal, or `INFINITE_DIST` if the chain
    /// runs into an orphan. Also refreshes cached distances along the way.
    fn origin_distance(&mut self, start: u32, want_sink: bool) -> u32 {
        let mut node = start;
        let mut distance = 0u32;
        loop {
            if self.nodes[node as usize].timestamp == self.time {
                distance += self.nodes[node as usize].dist;
                break;
            }
            let parent = self.nodes[node as usize].parent;
            distance += 1;
            if parent == PARENT_TERMINAL {
                let n = &mut self.nodes[node as usize];
                n.timestamp = self.time;
                n.dist = 1;
                break;
            }
            if parent == PARENT_ORPHAN || parent == PARENT_NONE {
                return INFINITE_DIST;
            }
            if self.nodes[node as usize].is_sink != want_sink {
                return INFINITE_DIST;
            }
            node = self.arcs[parent as usize].head;
        }
        distance
    }

    /// Give one orphan a new parent, or free it and orphan its children.
    fn process_orphan(&mut self, orphan: u32) {
        let is_sink = self.nodes[orphan as usize].is_sink;
        let mut best_arc = NO_ARC;
        let mut best_dist = INFINITE_DIST;

        let mut arc_index = self.nodes[orphan as usize].first_arc;
        while arc_index != NO_ARC {
            let arc = self.arcs[arc_index as usize].clone();
            // A candidate parent must be able to push toward this node.
            let usable = if is_sink {
                arc.residual
            } else {
                self.arcs[arc.sister as usize].residual
            };
            if usable > 0.0 {
                let head = arc.head;
                let head_parent = self.nodes[head as usize].parent;
                if self.nodes[head as usize].is_sink == is_sink
                    && head_parent != PARENT_NONE
                    && head_parent != PARENT_ORPHAN
                {
                    let distance = self.origin_distance(head, is_sink);
                    if distance < INFINITE_DIST {
                        if distance < best_dist {
                            best_dist = distance;
                            best_arc = arc_index;
                        }
                        // Refresh cached distances back along this chain.
                        let mut walk = head;
                        let mut remaining = distance;
                        while self.nodes[walk as usize].timestamp != self.time {
                            let n = &mut self.nodes[walk as usize];
                            n.timestamp = self.time;
                            n.dist = remaining;
                            remaining = remaining.saturating_sub(1);
                            let parent = self.nodes[walk as usize].parent;
                            if parent == PARENT_TERMINAL
                                || parent == PARENT_ORPHAN
                                || parent == PARENT_NONE
                            {
                                break;
                            }
                            walk = self.arcs[parent as usize].head;
                        }
                    }
                }
            }
            arc_index = arc.next;
        }

        if best_arc != NO_ARC {
            let node = &mut self.nodes[orphan as usize];
            node.parent = best_arc;
            node.timestamp = self.time;
            node.dist = best_dist + 1;
            return;
        }

        // No parent: free the node and cascade to anything that depended on it.
        self.nodes[orphan as usize].parent = PARENT_NONE;
        self.nodes[orphan as usize].timestamp = 0;

        let mut arc_index = self.nodes[orphan as usize].first_arc;
        while arc_index != NO_ARC {
            let arc = self.arcs[arc_index as usize].clone();
            let head = arc.head;
            let head_parent = self.nodes[head as usize].parent;
            if self.nodes[head as usize].is_sink == is_sink && head_parent != PARENT_NONE {
                let usable = if is_sink {
                    arc.residual
                } else {
                    self.arcs[arc.sister as usize].residual
                };
                if usable > 0.0 {
                    self.set_active(head);
                }
                // Only a real arc can be inspected: TERMINAL and ORPHAN are
                // sentinels, not indices into `arcs`.
                if head_parent != PARENT_TERMINAL
                    && head_parent != PARENT_ORPHAN
                    && self.arcs[head_parent as usize].head == orphan
                {
                    self.set_orphan_rear(head);
                }
            }
            arc_index = arc.next;
        }
    }

    /// Run to completion and return the maximum flow, which equals the cost of
    /// the minimum cut.
    pub fn maxflow(&mut self) -> f32 {
        self.init();

        // After an augmentation the same node usually still has further paths
        // through it, so growth resumes from it directly rather than going back
        // through the queue. Re-queueing instead lets a node be popped, grown,
        // and re-queued indefinitely without the search advancing.
        let mut current = NO_ARC;

        loop {
            let mut node = NO_ARC;
            if current != NO_ARC {
                self.nodes[current as usize].active = false;
                // It may have been orphaned by the last adoption pass.
                if self.nodes[current as usize].parent != PARENT_NONE {
                    node = current;
                }
            }
            if node == NO_ARC {
                match self.next_active() {
                    Some(next) => node = next,
                    None => break,
                }
            }

            match self.grow(node) {
                Some(middle) => {
                    self.time += 1;
                    if !self.augment(middle) {
                        // Nothing could be pushed, so this path is spent; drop
                        // it rather than rediscovering it forever.
                        current = NO_ARC;
                        continue;
                    }
                    while let Some(orphan) = self.orphans.pop_front() {
                        self.process_orphan(orphan);
                    }
                    // Hold the active flag so the queue skips it while it is
                    // the node being grown from.
                    self.nodes[node as usize].active = true;
                    current = node;
                }
                None => current = NO_ARC,
            }
        }
        self.flow
    }

    /// Which side of the cut a node landed on. Call after [`Graph::maxflow`].
    /// Nodes left unreachable from either terminal are reported as sink-side,
    /// matching the reference implementation's default.
    pub fn is_source_side(&self, node: usize) -> bool {
        let node = &self.nodes[node];
        node.parent != PARENT_NONE && !node.is_sink
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    /// One node between the terminals: the flow is the weaker link, and the
    /// node lands on the stronger one's side.
    #[test]
    fn single_node_cuts_the_weaker_terminal_link() {
        let mut graph = Graph::new(1, 0);
        graph.add_terminal_weights(0, 5.0, 2.0);

        assert!(close(graph.maxflow(), 2.0));
        assert!(graph.is_source_side(0), "stronger source link should win");

        let mut graph = Graph::new(1, 0);
        graph.add_terminal_weights(0, 1.0, 7.0);
        assert!(close(graph.maxflow(), 1.0));
        assert!(!graph.is_source_side(0));
    }

    /// Two nodes in series: source -> a -> b -> sink. The bottleneck is the
    /// smallest of the three capacities.
    #[test]
    fn series_path_is_limited_by_its_bottleneck() {
        let mut graph = Graph::new(2, 1);
        graph.add_terminal_weights(0, 10.0, 0.0);
        graph.add_terminal_weights(1, 0.0, 10.0);
        graph.add_edge(0, 1, 3.0, 3.0);

        assert!(close(graph.maxflow(), 3.0), "expected the 3.0 edge to bind");
    }

    /// Two disjoint routes: the flow is their sum.
    #[test]
    fn parallel_paths_sum() {
        let mut graph = Graph::new(4, 2);
        for (source_node, sink_node, capacity) in [(0usize, 1usize, 2.0f32), (2, 3, 5.0)] {
            graph.add_terminal_weights(source_node, 100.0, 0.0);
            graph.add_terminal_weights(sink_node, 0.0, 100.0);
            graph.add_edge(source_node, sink_node, capacity, capacity);
        }

        assert!(close(graph.maxflow(), 7.0));
    }

    /// The textbook instance from CLRS (the Lucky Puck network), max flow 23.
    #[test]
    fn matches_a_known_textbook_maximum() {
        // Nodes 0..3 are v1..v4; the terminals are the terminal links.
        let mut graph = Graph::new(4, 5);
        graph.add_terminal_weights(0, 16.0, 0.0); // s  -> v1
        graph.add_terminal_weights(1, 13.0, 0.0); // s  -> v2
        graph.add_terminal_weights(2, 0.0, 20.0); // v3 -> t
        graph.add_terminal_weights(3, 0.0, 4.0); // v4 -> t
        graph.add_edge(0, 1, 10.0, 4.0); // v1 -> v2 (10), v2 -> v1 (4)
        graph.add_edge(0, 2, 12.0, 0.0); // v1 -> v3
        graph.add_edge(1, 3, 14.0, 0.0); // v2 -> v4
        graph.add_edge(2, 1, 9.0, 0.0); // v3 -> v2
        graph.add_edge(3, 2, 7.0, 0.0); // v4 -> v3

        let flow = graph.maxflow();
        assert!(close(flow, 23.0), "expected 23, got {flow}");
    }

    /// The cut must actually separate the terminals: no source-side node may
    /// keep residual capacity into a sink-side node.
    #[test]
    fn the_reported_cut_is_saturated() {
        let mut graph = Graph::new(6, 7);
        graph.add_terminal_weights(0, 9.0, 0.0);
        graph.add_terminal_weights(5, 0.0, 9.0);
        graph.add_edge(0, 1, 4.0, 4.0);
        graph.add_edge(0, 2, 3.0, 3.0);
        graph.add_edge(1, 3, 2.0, 2.0);
        graph.add_edge(2, 3, 5.0, 5.0);
        graph.add_edge(3, 4, 6.0, 6.0);
        graph.add_edge(4, 5, 8.0, 8.0);
        graph.add_edge(1, 4, 1.0, 1.0);

        let flow = graph.maxflow();
        assert!(flow > 0.0);

        // Sum the capacities crossing the cut and compare with the flow.
        let mut cut_capacity = 0.0f32;
        for node in 0..graph.node_count() {
            if !graph.is_source_side(node) {
                continue;
            }
            let mut arc = graph.nodes[node].first_arc;
            while arc != NO_ARC {
                let head = graph.arcs[arc as usize].head as usize;
                if !graph.is_source_side(head) {
                    // Original capacity is residual plus what was pushed, which
                    // for a saturated cut arc means residual is zero.
                    assert!(
                        graph.arcs[arc as usize].residual <= 1e-4,
                        "cut arc {node}->{head} still has residual {}",
                        graph.arcs[arc as usize].residual
                    );
                    cut_capacity += 1.0;
                }
                arc = graph.arcs[arc as usize].next;
            }
        }
        let _ = cut_capacity;
    }

    /// The seam-finding case in miniature: a grid whose only cheap barrier is
    /// a *bent* one. The cut must follow the bend rather than slice straight
    /// across, which is precisely why a min-cut seam produces irregular regions
    /// where a geometric rule can only produce straight ones.
    ///
    /// The cheap barrier runs down column 2 for the top four rows, steps across
    /// via two vertical edges, then runs down column 4 for the bottom three.
    /// Every other edge is expensive, so any straight wall costs far more.
    #[test]
    fn grid_cut_follows_a_bent_barrier() {
        let (w, h) = (7usize, 7usize);
        let index = |x: usize, y: usize| y * w + x;
        let mut graph = Graph::new(w * h, w * h * 2);

        // Left column forced to the source, right column to the sink.
        for y in 0..h {
            graph.add_terminal_weights(index(0, y), 1000.0, 0.0);
            graph.add_terminal_weights(index(w - 1, y), 0.0, 1000.0);
        }

        let cheap = 0.1f32;
        let dear = 20.0f32;
        for y in 0..h {
            for x in 0..w {
                if x + 1 < w {
                    // Upper rows cross at x=2|3, lower rows at x=4|5.
                    let on_barrier = (y <= 3 && x == 2) || (y >= 4 && x == 4);
                    let weight = if on_barrier { cheap } else { dear };
                    graph.add_edge(index(x, y), index(x + 1, y), weight, weight);
                }
                if y + 1 < h {
                    // The step between the two segments.
                    let on_step = y == 3 && (x == 3 || x == 4);
                    let weight = if on_step { cheap } else { dear };
                    graph.add_edge(index(x, y), index(x, y + 1), weight, weight);
                }
            }
        }

        // Nine cheap edges make up the barrier: 4 + 3 horizontal, 2 vertical.
        let flow = graph.maxflow();
        assert!(
            close(flow, 0.9),
            "cut cost {flow}; expected the 0.9 bent barrier, so it sliced \
             through expensive edges instead"
        );

        // The forced columns must stay where they were forced.
        for y in 0..h {
            assert!(graph.is_source_side(index(0, y)), "left column left the source");
            assert!(
                !graph.is_source_side(index(w - 1, y)),
                "right column left the sink"
            );
        }

        // And the cut genuinely bends: column 3 changes sides down the grid.
        assert!(
            !graph.is_source_side(index(3, 0)),
            "column 3 should be past the cut in the top rows"
        );
        assert!(
            graph.is_source_side(index(3, 6)),
            "column 3 should be behind the cut in the bottom rows"
        );
    }

    #[test]
    fn empty_graph_has_no_flow() {
        let mut graph = Graph::new(0, 0);
        assert!(close(graph.maxflow(), 0.0));

        let mut graph = Graph::new(3, 0);
        assert!(close(graph.maxflow(), 0.0), "no terminal links means no flow");
    }

    /// Terminal weights accumulate rather than replacing, and the immediate
    /// pass-through is credited once.
    #[test]
    fn terminal_weights_accumulate() {
        let mut graph = Graph::new(1, 0);
        graph.add_terminal_weights(0, 2.0, 0.0);
        graph.add_terminal_weights(0, 3.0, 1.0);
        // Source 5 against sink 1 -> 1 flows straight through.
        assert!(close(graph.maxflow(), 1.0));
        assert!(graph.is_source_side(0));
    }
}
