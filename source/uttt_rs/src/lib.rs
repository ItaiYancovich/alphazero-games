//! Native Ultimate Tic-Tac-Toe core: rules, batched PUCT search, endgame solver.
//!
//! Built as a Python extension (`alphazero_uttt/uttt_rs.pyd`) by
//! `tools/build_uttt_rs.py`.  Python keeps the network; this keeps everything
//! the search does between network calls.

pub mod board;
pub mod search;
pub mod solver;

use board::{Board, ANY, FULL, WON};

/// Rebuild a position from each player's nine small-board masks.
pub fn board_from_masks(x: &[u16], o: &[u16], active: u8, stm: u8) -> Option<Board> {
    if x.len() != 9 || o.len() != 9 || stm > 1 || (active > 8 && active != ANY) {
        return None;
    }
    let mut b = Board::default();
    let mut marks = 0u32;
    for i in 0..9 {
        if x[i] & o[i] != 0 || x[i] > FULL || o[i] > FULL {
            return None;
        }
        b.small[0][i] = x[i];
        b.small[1][i] = o[i];
        marks += x[i].count_ones() + o[i].count_ones();
        if WON[x[i] as usize] {
            b.meta[0] |= 1 << i;
            b.closed |= 1 << i;
        } else if WON[o[i] as usize] {
            b.meta[1] |= 1 << i;
            b.closed |= 1 << i;
        } else if x[i] | o[i] == FULL {
            b.closed |= 1 << i;
        }
    }
    if WON[b.meta[0] as usize] {
        b.winner = 1;
    } else if WON[b.meta[1] as usize] {
        b.winner = 2;
    }
    b.stm = stm;
    b.moves = marks as u8;
    b.active = if active != ANY && (b.closed >> active) & 1 == 1 { ANY } else { active };
    Some(b)
}

#[cfg(feature = "python")]
mod py {
    use super::*;
    use crate::search::{solved_value, Config, MultiSearch, Tree, UNSOLVED};
    use crate::solver::{brute_force, Solver};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    #[pyclass(module = "uttt_rs")]
    pub struct Searcher {
        inner: MultiSearch,
    }

    #[pymethods]
    impl Searcher {
        #[new]
        #[pyo3(signature = (n_trees = 1, c_puct = 1.6, fpu_reduction = 0.25, solver_empties = 18,
                            solver_budget = 20000, cache_size = 300000, seed = 0,
                            variance_c_puct = false, q_init_weight = 0.0, draw_value = 0.0,
                            forced_playouts = 0.0))]
        fn new(n_trees: usize, c_puct: f32, fpu_reduction: f32, solver_empties: u32,
               solver_budget: u64, cache_size: usize, seed: u64, variance_c_puct: bool,
               q_init_weight: f32, draw_value: f32, forced_playouts: f32) -> Self {
            let cfg = Config { c_puct, fpu_reduction, solver_empties, solver_budget, cache_size,
                               variance_c_puct, q_init_weight, draw_value, forced_playouts };
            Searcher { inner: MultiSearch::new(n_trees, cfg, seed) }
        }

        fn n_trees(&self) -> usize {
            self.inner.trees.len()
        }

        fn set_root_moves(&mut self, t: usize, moves: Vec<u8>) -> PyResult<()> {
            let b = Board::from_moves(&moves).ok_or_else(|| PyValueError::new_err("illegal move sequence"))?;
            self.inner.trees[t] = Tree::new(b);
            Ok(())
        }

        fn set_root_state(&mut self, t: usize, x: Vec<u16>, o: Vec<u16>, active: u8, stm: u8) -> PyResult<()> {
            let b = board_from_masks(&x, &o, active, stm).ok_or_else(|| PyValueError::new_err("bad state"))?;
            self.inner.trees[t] = Tree::new(b);
            Ok(())
        }

        /// Re-root tree `t` at the move played, keeping the subtree.
        fn advance(&mut self, t: usize, mv: u8) -> PyResult<()> {
            if self.inner.trees[t].advance(mv) { Ok(()) } else { Err(PyValueError::new_err("illegal move")) }
        }

        #[pyo3(signature = (max_leaves, per_tree = 1, sims_limit = u32::MAX, max_collisions = 4))]
        fn collect<'py>(&mut self, py: Python<'py>, max_leaves: usize, per_tree: usize, sims_limit: u32,
                        max_collisions: usize) -> (Bound<'py, PyBytes>, Vec<u32>) {
            let n = self.inner.collect(max_leaves, per_tree, sims_limit, max_collisions);
            let mut buf = vec![0u8; n * 81];
            let mut ids = Vec::with_capacity(n);
            for (i, p) in self.inner.pending.iter().enumerate() {
                p.board.canonical(&mut buf[i * 81..i * 81 + 81]);
                ids.push(p.tree);
            }
            (PyBytes::new(py, &buf), ids)
        }

        /// Float32 little-endian priors [n*81], values [n] and optional per-move
        /// Q estimates [n*81], in `collect` order.
        #[pyo3(signature = (priors, values, qs = None))]
        fn apply(&mut self, priors: &[u8], values: &[u8], qs: Option<&[u8]>) -> PyResult<()> {
            let n = self.inner.pending.len();
            if priors.len() != n * 81 * 4 || values.len() != n * 4 {
                return Err(PyValueError::new_err(format!(
                    "expected {} priors and {} values for {} leaves", n * 81, n, n)));
            }
            if let Some(q) = qs {
                if q.len() != n * 81 * 4 {
                    return Err(PyValueError::new_err("qs must hold 81 floats per leaf"));
                }
            }
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            let p = f(priors);
            let v = f(values);
            let q = qs.map(f);
            self.inner.apply(&p, &v, q.as_deref());
            Ok(())
        }

        /// Per-tree cap on root visits used by `collect` (in addition to its `sims_limit`).
        fn set_limit(&mut self, t: usize, limit: u32) {
            self.inner.limits[t] = limit;
        }

        fn root_expanded(&self, t: usize) -> bool {
            self.inner.trees[t].root().state == 2
        }

        fn mix_root_noise(&mut self, t: usize, noise: Vec<f32>, eps: f32) -> bool {
            self.inner.trees[t].mix_root_noise(&noise, eps)
        }

        fn root_prior_temperature(&mut self, t: usize, temperature: f32) -> bool {
            self.inner.trees[t].root_prior_temperature(temperature)
        }

        fn root_n(&self, t: usize) -> u32 {
            self.inner.trees[t].root().n
        }

        /// Search value for the side to move at the root (a proven result if there is one).
        fn root_value(&self, t: usize) -> f32 {
            let r = self.inner.trees[t].root();
            if r.solved != UNSOLVED {
                return solved_value(r.solved, self.inner.cfg.draw_value);
            }
            if r.n == 0 { 0.0 } else { -(r.w / r.n as f64) as f32 }
        }

        fn root_solved(&self, t: usize) -> i8 {
            self.inner.trees[t].root().solved
        }

        /// (move, visits, q, prior, solved) for every root move, from the mover's view.
        /// `solved` is 1 won, -1 lost, 0 drawn, 2 unknown.
        fn root_children(&self, t: usize) -> Vec<(u8, u32, f32, f32, i8)> {
            let tree = &self.inner.trees[t];
            if tree.root().state != 2 {
                return Vec::new();
            }
            tree.children(0).iter().map(|c| {
                let q = if c.solved != UNSOLVED { -solved_value(c.solved, self.inner.cfg.draw_value) }
                        else if c.n > 0 { (c.w / c.n as f64) as f32 } else { f32::NAN };
                let s = if c.solved == UNSOLVED { UNSOLVED } else { -c.solved };
                (c.mv, c.n, q, c.prior, s)
            }).collect()
        }

        /// The root position as the network sees it (81 bytes, side-to-move
        /// relative), so a caller can evaluate a position it set up here.
        fn root_canonical<'py>(&self, py: Python<'py>, t: usize) -> Bound<'py, PyBytes> {
            let mut buf = [0u8; 81];
            self.inner.trees[t].root_board.canonical(&mut buf);
            PyBytes::new(py, &buf)
        }

        /// Cells still to play in boards that are still open -- what the exact
        /// solver's cost actually depends on, since closed boards can never be
        /// played in again.
        fn root_open_empties(&self, t: usize) -> u32 {
            self.inner.trees[t].root_board.open_empties()
        }

        /// Exact value of every root move, from the mover's view: 1 won, 0 drawn,
        /// -1 lost, 2 not settled within `budget` nodes.  See `MultiSearch::root_solve`.
        /// `seconds` caps the whole call on the wall clock (0: no cap).
        #[pyo3(signature = (t, budget, stop_on_win = false, seconds = 0.0))]
        fn root_solve(&mut self, py: Python<'_>, t: usize, budget: u64,
                      stop_on_win: bool, seconds: f64) -> Vec<(u8, i8)> {
            // Seconds of pure Rust with no Python touched: let the GUI's other
            // threads run rather than freezing the page while we think.
            py.detach(|| self.inner.root_solve(t, budget, stop_on_win, seconds))
        }

        /// Principal variation: most-visited child at each step.
        fn pv(&self, t: usize, max_len: usize) -> Vec<u8> {
            let tree = &self.inner.trees[t];
            let mut out = Vec::new();
            let mut idx = 0usize;
            while out.len() < max_len && tree.nodes[idx].state == 2 && tree.nodes[idx].n_children > 0 {
                let first = tree.nodes[idx].first_child as usize;
                let k = tree.nodes[idx].n_children as usize;
                let best = (first..first + k).max_by_key(|&i| {
                    let c = &tree.nodes[i];
                    (if c.solved != UNSOLVED && -c.solved == 1 { 1 } else { 0 }, c.n)
                }).unwrap();
                if tree.nodes[best].n == 0 {
                    break;
                }
                out.push(tree.nodes[best].mv);
                idx = best;
            }
            out
        }

        fn node_count(&self, t: usize) -> usize {
            self.inner.trees[t].nodes.len()
        }

        /// (cache hits, solver calls, solver successes)
        fn stats(&self) -> (u64, u64, u64) {
            (self.inner.cache_hits, self.inner.solver_calls, self.inner.solver_solved)
        }

        fn clear_cache(&mut self) {
            self.inner.clear_cache();
        }

        fn set_variance_c_puct(&mut self, on: bool) {
            self.inner.cfg.variance_c_puct = on;
        }

        fn set_q_init_weight(&mut self, w: f32) {
            self.inner.cfg.q_init_weight = w;
        }

        /// (move, visits) for the policy target, forced playouts subtracted out.
        fn root_policy_target(&self, t: usize, k: f32) -> Vec<(u8, f32)> {
            self.inner.trees[t].root_policy_target(k)
        }

        fn set_draw_value(&mut self, v: f32) {
            self.inner.cfg.draw_value = v;
        }

        fn set_c_puct(&mut self, c: f32) {
            self.inner.cfg.c_puct = c;
        }

        fn set_fpu_reduction(&mut self, f: f32) {
            self.inner.cfg.fpu_reduction = f;
        }
    }

    /// Replay moves; returns (legal moves, terminal, winner 0/1/2), or None if illegal.
    #[pyfunction]
    fn legal_moves(moves: Vec<u8>) -> Option<(Vec<u8>, bool, u8)> {
        let b = Board::from_moves(&moves)?;
        let mut buf = [0u8; 81];
        let n = b.legal(&mut buf);
        Some((buf[..n].to_vec(), b.is_terminal(), b.winner))
    }

    /// Exact result for the side to move after `moves`, or None over budget.
    #[pyfunction]
    fn solve(moves: Vec<u8>, budget: u64) -> PyResult<Option<i8>> {
        let b = Board::from_moves(&moves).ok_or_else(|| PyValueError::new_err("illegal move sequence"))?;
        Ok(Solver::new().solve(&b, budget))
    }

    #[pyfunction]
    fn solve_state(x: Vec<u16>, o: Vec<u16>, active: u8, stm: u8, budget: u64) -> PyResult<Option<i8>> {
        let b = board_from_masks(&x, &o, active, stm).ok_or_else(|| PyValueError::new_err("bad state"))?;
        Ok(Solver::new().solve(&b, budget))
    }

    #[pyfunction]
    fn brute_force_state(x: Vec<u16>, o: Vec<u16>, active: u8, stm: u8) -> PyResult<i8> {
        let b = board_from_masks(&x, &o, active, stm).ok_or_else(|| PyValueError::new_err("bad state"))?;
        Ok(brute_force(&b))
    }

    /// Canonical 81-cell board after `moves` (0 playable, 1 own, 2 opp, 3 unreachable).
    #[pyfunction]
    fn canonical<'py>(py: Python<'py>, moves: Vec<u8>) -> PyResult<Bound<'py, PyBytes>> {
        let b = Board::from_moves(&moves).ok_or_else(|| PyValueError::new_err("illegal move sequence"))?;
        let mut buf = [0u8; 81];
        b.canonical(&mut buf);
        Ok(PyBytes::new(py, &buf))
    }

    /// Random playouts for `seconds`: (games, plies, elapsed).
    #[pyfunction]
    fn bench_playouts(seconds: f64, seed: u64) -> (u64, u64, f64) {
        let mut rng = board::Rng(seed | 1);
        let t0 = std::time::Instant::now();
        let (mut games, mut plies) = (0u64, 0u64);
        let mut buf = [0u8; 81];
        while t0.elapsed().as_secs_f64() < seconds {
            for _ in 0..256 {
                let mut b = Board::default();
                loop {
                    let n = b.legal(&mut buf);
                    if n == 0 {
                        break;
                    }
                    b.play(buf[rng.below(n)]);
                    plies += 1;
                }
                games += 1;
            }
        }
        (games, plies, t0.elapsed().as_secs_f64())
    }

    #[pymodule]
    fn uttt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Searcher>()?;
        m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
        m.add_function(wrap_pyfunction!(solve, m)?)?;
        m.add_function(wrap_pyfunction!(solve_state, m)?)?;
        m.add_function(wrap_pyfunction!(brute_force_state, m)?)?;
        m.add_function(wrap_pyfunction!(canonical, m)?)?;
        m.add_function(wrap_pyfunction!(bench_playouts, m)?)?;
        Ok(())
    }
}
