//! Exact endgame solver: negamax alpha-beta over the three results {-1, 0, +1},
//! with a transposition table and a node budget.
//!
//! Late in a game few cells are left in the open boards, and the tree is small
//! enough to settle exactly.  That is where a strong human punishes a bot that
//! only estimates, so the search calls this on leaves below an empties
//! threshold and treats a solved leaf as a terminal.

use crate::board::{Board, BOARD_OF, SLOT_OF, WON};
use std::collections::HashMap;
use std::time::Instant;

const EXACT: u8 = 0;
const LOWER: u8 = 1;
const UPPER: u8 = 2;

pub struct Solver {
    tt: HashMap<u64, (i8, u8)>,
    pub nodes: u64,
    budget: u64,
    pub max_tt: usize,
    /// Wall-clock cutoff, checked every few thousand nodes.  A node budget
    /// alone is no use against a clock: 40M nodes took 10 s in one real
    /// position and 57 s in another.
    pub deadline: Option<Instant>,
}

impl Solver {
    pub fn new() -> Self {
        Solver { tt: HashMap::with_capacity(1 << 16), nodes: 0, budget: 0, max_tt: 4_000_000,
                 deadline: None }
    }

    /// Exact value for the side to move, or None if the budget ran out.
    pub fn solve(&mut self, b: &Board, budget: u64) -> Option<i8> {
        self.solve_window(b, -1, 1, budget)
    }

    /// The same, inside a window.  Outside `[alpha, beta]` the answer comes back
    /// as a bound rather than the exact value, which is all a caller comparing
    /// against a known best needs -- and is far cheaper, because the cutoff can
    /// fire at once.  Solving every root move with a full window instead costs
    /// about a hundred times as much (measured: 116 s against 1 s at ply 36).
    pub fn solve_window(&mut self, b: &Board, alpha: i8, beta: i8, budget: u64) -> Option<i8> {
        // A window with no width is not a question alpha-beta can answer: it
        // cuts off at once and would store a meaningless bound as a real one.
        assert!(alpha < beta, "solve_window needs alpha < beta, got [{alpha}, {beta}]");
        self.nodes = 0;
        self.budget = budget;
        if self.tt.len() > self.max_tt {
            self.tt.clear();
        }
        self.negamax(b, alpha, beta)
    }

    fn order(&self, b: &Board, moves: &mut [u8], n: usize) -> usize {
        // Score: win the game > win a small board > quiet; sending the
        // opponent to a free choice or to a board they can close sorts last.
        let p = b.stm as usize;
        let o = 1 - p;
        let mut scored: [(i32, u8); 81] = [(0, 0); 81];
        for i in 0..n {
            let c = moves[i];
            let bd = BOARD_OF[c as usize] as usize;
            let k = SLOT_OF[c as usize] as usize;
            let mine = b.small[p][bd] | (1 << k);
            let mut s = 0;
            if WON[mine as usize] {
                if WON[(b.meta[p] | (1 << bd)) as usize] {
                    return i + 1000; // signal: game-winning move at index i
                }
                s += 50;
            }
            // Destination after this move.
            let closed_after = if WON[mine as usize] || (mine | b.small[o][bd]) == 0x1FF {
                b.closed | (1 << bd)
            } else {
                b.closed
            };
            if (closed_after >> k) & 1 == 1 {
                s -= 30;
            } else {
                let occ = b.small[0][k] | b.small[1][k] | if k == bd { 1 << k } else { 0 };
                let opp = b.small[o][k];
                let mut free = !occ & 0x1FF;
                while free != 0 {
                    let j = free.trailing_zeros();
                    if WON[(opp | (1 << j)) as usize] {
                        s -= 20;
                        break;
                    }
                    free &= free - 1;
                }
            }
            scored[i] = (s, c);
        }
        scored[..n].sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for i in 0..n {
            moves[i] = scored[i].1;
        }
        usize::MAX
    }

    fn negamax(&mut self, b: &Board, mut alpha: i8, mut beta: i8) -> Option<i8> {
        if b.is_terminal() {
            return Some(b.terminal_value() as i8);
        }
        self.nodes += 1;
        if self.nodes > self.budget {
            return None;
        }
        if self.nodes & 4095 == 0 {
            if let Some(d) = self.deadline {
                if Instant::now() >= d {
                    return None;
                }
            }
        }
        let key = b.key();
        let a0 = alpha;
        if let Some(&(v, flag)) = self.tt.get(&key) {
            match flag {
                EXACT => return Some(v),
                LOWER => alpha = alpha.max(v),
                _ => beta = beta.min(v),
            }
            if alpha >= beta {
                return Some(v);
            }
        }
        let mut moves = [0u8; 81];
        let n = b.legal(&mut moves);
        let w = self.order(b, &mut moves, n);
        if w != usize::MAX {
            return Some(1);
        }
        let mut best: i8 = -2;
        for i in 0..n {
            let mut c = *b;
            c.play(moves[i]);
            let v = -self.negamax(&c, -beta, -alpha)?;
            if v > best {
                best = v;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }
        let flag = if best <= a0 { UPPER } else if best >= beta { LOWER } else { EXACT };
        self.tt.insert(key, (best, flag));
        Some(best)
    }
}

/// Plain minimax with no pruning and no table, for testing the solver.
pub fn brute_force(b: &Board) -> i8 {
    if b.is_terminal() {
        return b.terminal_value() as i8;
    }
    let mut moves = [0u8; 81];
    let n = b.legal(&mut moves);
    let mut best = -2;
    for &m in &moves[..n] {
        let mut c = *b;
        c.play(m);
        best = best.max(-brute_force(&c));
        if best == 1 {
            break;
        }
    }
    best
}
