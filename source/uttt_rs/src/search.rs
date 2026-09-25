//! Batched PUCT search with virtual loss, MCTS-Solver, an exact endgame solver
//! at the leaves, an evaluation cache and tree reuse.
//!
//! The network lives in Python.  `collect` walks the trees until it has a batch
//! of leaves that need an evaluation and returns their canonical boards;
//! `apply` takes the priors and values back and expands.  Everything between
//! those two calls -- selection, rules, proving wins, backing up -- is here.
//!
//! Perspective: `Node::w` sums values from the point of view of the player who
//! made the move *into* the node.  `Node::solved` is from the point of view of
//! the player *to move at* the node.

use crate::board::{Board, Rng};
use crate::solver::Solver;
use std::collections::HashMap;

pub const UNSOLVED: i8 = 2;

/// Value of a solved flag from the point of view of the player it belongs to.
#[inline]
pub fn solved_value(flag: i8, draw_value: f32) -> f32 {
    match flag {
        0 => draw_value,
        v => v as f32,
    }
}
const UNEXPANDED: u8 = 0;
const PENDING: u8 = 1;
const EXPANDED: u8 = 2;

#[derive(Clone, Copy)]
pub struct Node {
    pub parent: u32,
    pub first_child: u32,
    pub n_children: u8,
    pub mv: u8,
    pub solved: i8,
    pub state: u8,
    pub prior: f32,
    pub n: u32,
    pub vl: u32,
    pub w: f64,
    pub w2: f32,        // sum of squared values, for variance-scaled cPUCT
    pub q_init: f32,    // the Q head's estimate for this move; NaN when unknown
}

impl Node {
    fn new(parent: u32, mv: u8, prior: f32) -> Self {
        Node { parent, first_child: 0, n_children: 0, mv, solved: UNSOLVED, state: UNEXPANDED,
               prior, n: 0, vl: 0, w: 0.0, w2: 0.0, q_init: f32::NAN }
    }
}

#[derive(Clone, Copy)]
pub struct Config {
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub solver_empties: u32,
    pub solver_budget: u64,
    pub cache_size: usize,
    /// KataGo's variance-scaled exploration: cPUCT rises where the value is noisy.
    pub variance_c_puct: bool,
    /// Weight of the Q head's per-move estimate as the starting value of an
    /// unvisited move; 0 falls back to first-play urgency.
    pub q_init_weight: f32,
    /// What a draw is worth to the side to move.  0 is the truth and is what
    /// self-play uses; a small negative value is contempt -- it makes the bot
    /// prefer a sharp position to a dead one, which is how you beat a human who
    /// is happy to draw.  It never changes which moves are legal or which wins
    /// are proven, only how a draw is scored.
    pub draw_value: f32,
    /// KataGo's forced playouts: every root move is guaranteed
    /// `sqrt(k * prior * N)` visits, and exactly those are subtracted back out
    /// of the policy target, so exploration happens without teaching the net
    /// the noise that caused it.  0 turns it off, which is what match play wants.
    pub forced_playouts: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config { c_puct: 1.6, fpu_reduction: 0.25, solver_empties: 18, solver_budget: 20_000,
                 cache_size: 300_000, variance_c_puct: false, q_init_weight: 0.0,
                 draw_value: 0.0, forced_playouts: 0.0 }
    }
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub root_board: Board,
}

impl Tree {
    pub fn new(b: Board) -> Self {
        Tree { nodes: vec![Node::new(u32::MAX, 255, 1.0)], root_board: b }
    }

    pub fn root(&self) -> &Node {
        &self.nodes[0]
    }

    pub fn children(&self, idx: usize) -> &[Node] {
        let n = &self.nodes[idx];
        let s = n.first_child as usize;
        &self.nodes[s..s + n.n_children as usize]
    }

    /// Which child of `idx` to descend into (PUCT with first-play urgency).
    fn select_child(&self, idx: usize, cfg: &Config) -> usize {
        let node = &self.nodes[idx];
        if idx == 0 && cfg.forced_playouts > 0.0 {
            // Any root move short of its guaranteed visits is taken first.
            let n_root = (node.n + node.vl).max(1) as f32;
            let first = node.first_child as usize;
            for i in first..first + node.n_children as usize {
                let c = &self.nodes[i];
                if c.solved != UNSOLVED {
                    continue;
                }
                let forced = (cfg.forced_playouts * c.prior * n_root).sqrt();
                if ((c.n + c.vl) as f32) < forced {
                    return i;
                }
            }
        }
        let first = node.first_child as usize;
        let k = node.n_children as usize;
        let n_parent = (node.n + node.vl).max(1) as f32;
        let sqrt_n = n_parent.sqrt();
        let q_parent = if node.n > 0 { -(node.w / node.n as f64) as f32 } else { 0.0 };
        let mut visited_prior = 0.0f32;
        for c in &self.nodes[first..first + k] {
            if c.n + c.vl > 0 {
                visited_prior += c.prior;
            }
        }
        let fpu = q_parent - cfg.fpu_reduction * visited_prior.sqrt();
        let c_eff = if cfg.variance_c_puct && node.n > 8 {
            // Variance of the values backed up through this node, against the
            // 0.25 of a fair coin on +/-1; clamped so it stays a nudge.
            let mean = node.w / node.n as f64;
            let var = (node.w2 as f64 / node.n as f64 - mean * mean).max(0.0);
            cfg.c_puct * (0.5 + 0.5 * (var / 0.25).sqrt() as f32).clamp(0.6, 1.8)
        } else {
            cfg.c_puct
        };
        let mut best = first;
        let mut best_score = f32::NEG_INFINITY;
        for i in first..first + k {
            let c = &self.nodes[i];
            if c.solved != UNSOLVED && c.solved != 0 {
                let v = -c.solved; // chooser's view
                if v == 1 {
                    return i;
                }
                if v == -1 {
                    let s = -1e9 + c.prior; // only if everything else is lost too
                    if s > best_score {
                        best_score = s;
                        best = i;
                    }
                    continue;
                }
            }
            let ne = c.n + c.vl;
            let q = if c.solved == 0 {
                cfg.draw_value
            } else if ne > 0 {
                ((c.w - c.vl as f64) / ne as f64) as f32
            } else if cfg.q_init_weight > 0.0 && c.q_init.is_finite() {
                let w = cfg.q_init_weight;
                w * c.q_init + (1.0 - w) * fpu
            } else {
                fpu
            };
            let u = c_eff * c.prior * sqrt_n / (1.0 + ne as f32);
            let s = q + u;
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        best
    }

    /// Add `v` (from the point of view of the player to move at `leaf`) up the path.
    fn backup(&mut self, leaf: usize, v: f32, remove_vl: bool) {
        let mut idx = leaf;
        let mut vm = v as f64;
        loop {
            let node = &mut self.nodes[idx];
            node.n += 1;
            node.w += -vm;
            node.w2 += (vm * vm) as f32;
            if remove_vl && node.vl > 0 {
                node.vl -= 1;
            }
            if node.parent == u32::MAX {
                break;
            }
            idx = node.parent as usize;
            vm = -vm;
        }
    }

    fn add_vl(&mut self, leaf: usize) {
        let mut idx = leaf;
        loop {
            self.nodes[idx].vl += 1;
            let p = self.nodes[idx].parent;
            if p == u32::MAX {
                break;
            }
            idx = p as usize;
        }
    }

    /// After `idx` changed, settle solved states upward.
    fn propagate_solved(&mut self, mut idx: usize) {
        loop {
            let p = self.nodes[idx].parent;
            if p == u32::MAX {
                break;
            }
            let pi = p as usize;
            if self.nodes[pi].solved != UNSOLVED || self.nodes[pi].state != EXPANDED {
                break;
            }
            let mut all = true;
            let mut best = -2i8;
            for c in self.children(pi) {
                if c.solved == UNSOLVED {
                    all = false;
                } else {
                    best = best.max(-c.solved);
                }
            }
            if best == 1 {
                self.nodes[pi].solved = 1;
            } else if all {
                self.nodes[pi].solved = best;
            } else {
                break;
            }
            idx = pi;
        }
    }

    fn expand(&mut self, idx: usize, board: &Board, priors: &[f32]) {
        self.expand_with_q(idx, board, priors, None)
    }

    fn expand_with_q(&mut self, idx: usize, board: &Board, priors: &[f32], qs: Option<&[f32]>) {
        let mut moves = [0u8; 81];
        let n = board.legal(&mut moves);
        let mut total = 0.0f32;
        for &m in &moves[..n] {
            total += priors[m as usize].max(0.0);
        }
        let first = self.nodes.len() as u32;
        for &m in &moves[..n] {
            let p = if total > 1e-12 { priors[m as usize].max(0.0) / total } else { 1.0 / n as f32 };
            let mut child = Node::new(idx as u32, m, p);
            if let Some(q) = qs {
                child.q_init = q[m as usize];
            }
            self.nodes.push(child);
        }
        let node = &mut self.nodes[idx];
        node.first_child = first;
        node.n_children = n as u8;
        node.state = EXPANDED;
    }

    /// Visit counts for the policy target, with the forced playouts taken back
    /// out: every root move except the most-visited one loses up to the visits it
    /// was guaranteed, so what is learned is what the search actually preferred.
    pub fn root_policy_target(&self, k: f32) -> Vec<(u8, f32)> {
        let root = self.nodes[0];
        if root.state != EXPANDED || root.n_children == 0 {
            return Vec::new();
        }
        let first = root.first_child as usize;
        let kids = first..first + root.n_children as usize;
        // First maximum, not the last: ties must resolve the same way everywhere.
        let best = kids.clone().fold(first, |b, i| if self.nodes[i].n > self.nodes[b].n { i } else { b });
        let n_root = root.n.max(1) as f32;
        let mut out = Vec::with_capacity(root.n_children as usize);
        let mut total = 0.0f32;
        for i in kids {
            let c = &self.nodes[i];
            let mut n = c.n as f32;
            if k > 0.0 && i != best {
                let forced = (k * c.prior * n_root).sqrt();
                n = (n - forced).max(0.0);
            }
            total += n;
            out.push((c.mv, n));
        }
        if total <= 0.0 {
            // Everything was pruned away (a very short search): fall back to raw visits.
            out.clear();
            for i in first..first + root.n_children as usize {
                out.push((self.nodes[i].mv, self.nodes[i].n as f32));
            }
        }
        out
    }

    /// Blend root priors with `noise` (one entry per root child, in child order).
    pub fn mix_root_noise(&mut self, noise: &[f32], eps: f32) -> bool {
        let root = self.nodes[0];
        if root.state != EXPANDED || noise.len() != root.n_children as usize {
            return false;
        }
        for (i, z) in noise.iter().enumerate() {
            let c = &mut self.nodes[root.first_child as usize + i];
            c.prior = (1.0 - eps) * c.prior + eps * z;
        }
        true
    }

    /// Sharpen (<1) or flatten (>1) root priors: p^(1/temperature), renormalised.
    pub fn root_prior_temperature(&mut self, temperature: f32) -> bool {
        let root = self.nodes[0];
        if root.state != EXPANDED || root.n_children == 0 {
            return false;
        }
        let first = root.first_child as usize;
        let k = root.n_children as usize;
        let mut total = 0.0f32;
        for i in first..first + k {
            let p = self.nodes[i].prior.max(1e-12).powf(1.0 / temperature);
            self.nodes[i].prior = p;
            total += p;
        }
        for i in first..first + k {
            self.nodes[i].prior /= total;
        }
        true
    }

    /// Re-root at the child played by `mv`, keeping its subtree.
    pub fn advance(&mut self, mv: u8) -> bool {
        let mut nb = self.root_board;
        if !nb.is_legal(mv) {
            return false;
        }
        nb.play(mv);
        let root = &self.nodes[0];
        let mut found = None;
        if root.state == EXPANDED {
            for i in 0..root.n_children as usize {
                let ci = root.first_child as usize + i;
                if self.nodes[ci].mv == mv {
                    found = Some(ci);
                }
            }
        }
        let Some(ci) = found else {
            *self = Tree::new(nb);
            return true;
        };
        // Breadth-first copy of the subtree under `ci`.
        let mut new_nodes: Vec<Node> = Vec::with_capacity(self.nodes.len() / 2);
        let mut rootn = self.nodes[ci];
        rootn.parent = u32::MAX;
        rootn.vl = 0;
        if rootn.state != EXPANDED {
            // A leaf the solver settled without expanding: start it afresh so the
            // agent gets children to choose between (they re-solve one level down).
            rootn = Node::new(u32::MAX, rootn.mv, 1.0);
        }
        new_nodes.push(rootn);
        let mut queue: Vec<(usize, usize)> = vec![(ci, 0)];
        let mut head = 0;
        while head < queue.len() {
            let (old, new) = queue[head];
            head += 1;
            let o = self.nodes[old];
            if o.state != EXPANDED {
                continue;
            }
            let first_new = new_nodes.len() as u32;
            for j in 0..o.n_children as usize {
                let oc = o.first_child as usize + j;
                let mut c = self.nodes[oc];
                c.parent = new as u32;
                c.vl = 0;
                if c.state == PENDING {
                    c.state = UNEXPANDED;
                }
                new_nodes.push(c);
                queue.push((oc, first_new as usize + j));
            }
            new_nodes[new].first_child = first_new;
        }
        self.nodes = new_nodes;
        self.root_board = nb;
        true
    }
}

pub struct Pending {
    pub tree: u32,
    pub node: u32,
    pub key: u64,
    pub board: Board,
}

pub struct MultiSearch {
    pub trees: Vec<Tree>,
    pub limits: Vec<u32>,
    pub cfg: Config,
    pub pending: Vec<Pending>,
    solver: Solver,
    cache: HashMap<u64, ([f32; 81], f32, Option<[f32; 81]>)>,
    pub cache_hits: u64,
    pub solver_calls: u64,
    pub solver_solved: u64,
    pub rng: Rng,
}

pub enum Step {
    Leaf,
    Done,
    Collision,
}

impl MultiSearch {
    pub fn new(n_trees: usize, cfg: Config, seed: u64) -> Self {
        MultiSearch {
            trees: (0..n_trees).map(|_| Tree::new(Board::default())).collect(),
            limits: vec![u32::MAX; n_trees],
            cfg,
            pending: Vec::new(),
            solver: Solver::new(),
            cache: HashMap::new(),
            cache_hits: 0,
            solver_calls: 0,
            solver_solved: 0,
            rng: Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1),
        }
    }

    /// One simulation's descent in tree `t`.
    pub fn step(&mut self, t: usize) -> Step {
        let cfg = self.cfg;
        let tree = &mut self.trees[t];
        let mut board = tree.root_board;
        let mut idx = 0usize;
        loop {
            let node = tree.nodes[idx];
            if node.solved != UNSOLVED && !(idx == 0 && node.state != EXPANDED) {
                tree.backup(idx, solved_value(node.solved, cfg.draw_value), false);
                return Step::Done;
            }
            if board.is_terminal() {
                let v = board.terminal_value() as i8;
                tree.nodes[idx].solved = v;
                tree.backup(idx, solved_value(v, cfg.draw_value), false);
                tree.propagate_solved(idx);
                return Step::Done;
            }
            match node.state {
                UNEXPANDED => {
                    // Never solve the root in place: the agent needs its children
                    // to pick a move, and they get solved one level down instead.
                    if idx != 0 && cfg.solver_empties > 0 && board.open_empties() <= cfg.solver_empties {
                        self.solver_calls += 1;
                        if let Some(v) = self.solver.solve(&board, cfg.solver_budget) {
                            self.solver_solved += 1;
                            tree.nodes[idx].solved = v;
                            tree.backup(idx, solved_value(v, cfg.draw_value), false);
                            tree.propagate_solved(idx);
                            return Step::Done;
                        }
                    }
                    let key = board.key();
                    if let Some((pri, v, qs)) = self.cache.get(&key) {
                        self.cache_hits += 1;
                        let (pri, v, qs) = (*pri, *v, *qs);
                        tree.expand_with_q(idx, &board, &pri, qs.as_ref().map(|a| &a[..]));
                        tree.backup(idx, v, false);
                        return Step::Done;
                    }
                    tree.nodes[idx].state = PENDING;
                    tree.add_vl(idx);
                    self.pending.push(Pending { tree: t as u32, node: idx as u32, key, board });
                    return Step::Leaf;
                }
                PENDING => return Step::Collision,
                _ => {
                    let c = tree.select_child(idx, &cfg);
                    board.play(tree.nodes[c].mv);
                    idx = c;
                }
            }
        }
    }

    /// Settle the root position itself, exactly, one move at a time.
    ///
    /// The per-leaf solver only fires deep in the tree, where the position is
    /// nearly over.  This asks the same question of the move actually about to
    /// be played, which real games reach with around fifteen plies still to go
    /// -- the last third of the game stops being estimated and starts being
    /// known.  Returns one entry per legal root move: its exact value from
    /// *our* side (+1 win, 0 draw, -1 loss), or 2 for a move the budget could
    /// not settle.  `budget` is nodes per move, spent afresh on each.
    ///
    /// `stop_on_win` is accepted but no longer changes anything: the first pass
    /// always stops at the first win, and classifying the rest is cheap.  The
    /// solver's transposition table
    /// is shared with the in-tree calls and survives between moves, so a later
    /// solve inherits everything earlier ones learned.
    pub fn root_solve(&mut self, t: usize, budget: u64, stop_on_win: bool,
                      seconds: f64) -> Vec<(u8, i8)> {
        // One deadline for the whole call, both passes; 0 means none.
        self.solver.deadline = if seconds > 0.0 {
            Some(std::time::Instant::now() + std::time::Duration::from_secs_f64(seconds))
        } else {
            None
        };
        let out = self.root_solve_inner(t, budget, stop_on_win);
        self.solver.deadline = None;    // the in-tree solver calls have no clock
        out
    }

    fn root_solve_inner(&mut self, t: usize, budget: u64, _stop_on_win: bool) -> Vec<(u8, i8)> {
        let board = self.trees[t].root_board;
        let mut moves = [0u8; 81];
        let n = board.legal(&mut moves);

        // First pass: the best value available, with the window tightening as it
        // rises, so inferior moves are cut off instead of being solved.
        let mut best: i8 = -2;
        let mut alpha: i8 = -1;
        let mut ran_out = false;
        for &mv in &moves[..n] {
            let mut child = board;
            child.play(mv);
            let v = if child.is_terminal() {
                -(child.terminal_value() as i8)
            } else {
                match self.solver.solve_window(&child, -1, -alpha, budget) {
                    Some(v) => -v,
                    None => {
                        ran_out = true;
                        continue;
                    }
                }
            };
            if v > best {
                best = v;
                alpha = alpha.max(v);
            }
            if best == 1 {
                // Nothing beats a win, and going on would search the next move
                // with the window (-1, -1): zero width, which cuts off after the
                // first reply and stores that trivial lower bound in the table
                // as an upper one -- "lost" for a position that is not.  That
                // poisoned later lookups (test_root_solve_agrees_with_brute_force
                // _move_by_move).  The second pass classifies every move with a
                // proper width-one window instead.
                break;
            }
        }

        if best == -2 || (ran_out && best < 1) {
            // Nothing settled, or something was left unsolved that might beat
            // what we found.  Report the position as unsettled rather than
            // acting on a "best" that only holds over part of the move list.
            return moves[..n].iter().map(|&m| (m, 2i8)).collect();
        }

        // Second pass: which moves actually reach that value.  Each is a single
        // bound test against `best`, and the table is warm from the first pass,
        // so this is nearly free.  Anything short of `best` is reported as -2:
        // its exact value is not worth paying for, because we would never play
        // it over a move that reaches `best`.
        let mut out = Vec::with_capacity(n);
        for &mv in &moves[..n] {
            let mut child = board;
            child.play(mv);
            let v = if child.is_terminal() {
                if -(child.terminal_value() as i8) >= best { best } else { -2 }
            } else {
                match self.solver.solve_window(&child, -best, -best + 1, budget) {
                    Some(v) if -v >= best => best,
                    Some(_) => -2,
                    None => 2,
                }
            };
            out.push((mv, v));
        }
        out
    }

    /// Gather up to `max_leaves` leaves, at most `per_tree` from each tree whose
    /// root has fewer than `sims_limit` visits.  Returns how many leaves.
    pub fn collect(&mut self, max_leaves: usize, per_tree: usize, sims_limit: u32,
                   max_collisions: usize) -> usize {
        self.pending.clear();
        let n_trees = self.trees.len();
        for t in 0..n_trees {
            if self.pending.len() >= max_leaves {
                break;
            }
            let mut got = 0;
            let mut collisions = 0;
            let mut guard = 0;
            while got < per_tree && self.pending.len() < max_leaves {
                let root = self.trees[t].root();
                if root.n + root.vl >= sims_limit.min(self.limits[t])
                    || (root.solved != UNSOLVED && root.state == EXPANDED) {
                    break;
                }
                guard += 1;
                if guard > per_tree * 8 + 64 {
                    break;
                }
                match self.step(t) {
                    Step::Leaf => got += 1,
                    Step::Done => {}
                    Step::Collision => {
                        collisions += 1;
                        if collisions > max_collisions {
                            break;
                        }
                    }
                }
            }
        }
        self.pending.len()
    }

    /// `qs`, when given, is the Q head's value for every cell of every leaf, in
    /// the same order as `collect` returned them.
    pub fn apply(&mut self, priors: &[f32], values: &[f32], qs: Option<&[f32]>) {
        let pend = std::mem::take(&mut self.pending);
        for (i, p) in pend.iter().enumerate() {
            let pri = &priors[i * 81..i * 81 + 81];
            let q = qs.map(|q| &q[i * 81..i * 81 + 81]);
            let v = values[i];
            let tree = &mut self.trees[p.tree as usize];
            let idx = p.node as usize;
            tree.expand_with_q(idx, &p.board, pri, q);
            tree.backup(idx, v, true);
            if self.cfg.cache_size > 0 {
                if self.cache.len() >= self.cfg.cache_size {
                    self.cache.clear();
                }
                let mut arr = [0f32; 81];
                arr.copy_from_slice(pri);
                let qarr = q.map(|q| {
                    let mut a = [0f32; 81];
                    a.copy_from_slice(q);
                    a
                });
                self.cache.insert(p.key, (arr, v, qarr));
            }
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}
