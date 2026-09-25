//! `uttt_rs` for the browser: the same search core, behind a plain C ABI.
//!
//! The desktop build wraps `uttt_rs` in PyO3.  Pyodide cannot load that, so
//! this crate compiles the identical core to WebAssembly (`wasm32-wasip1`: the
//! core times its solver with `Instant`, which needs a clock) and exposes it
//! as flat functions over linear memory.  `webapp/py/alphazero_uttt_rs_web.py`
//! turns them back into the `uttt_rs` module the Python agent imports, method
//! for method -- so the search, the solver and the agent are unchanged.
//!
//! Conventions: a searcher is a small integer handle.  Inputs larger than a
//! scalar are copied into memory from `alloc` first; results larger than a
//! scalar are left in one shared output buffer (`out_ptr`/`out_len`) that the
//! next call overwrites.  64-bit counts travel as `f64`, which JavaScript
//! passes without BigInt.

use std::cell::RefCell;

use uttt_rs::board::Board;
use uttt_rs::board_from_masks;
use uttt_rs::search::{solved_value, Config, MultiSearch, Tree, UNSOLVED};
use uttt_rs::solver::{brute_force, Solver};

thread_local! {
    static SEARCHERS: RefCell<Vec<Option<MultiSearch>>> = RefCell::new(Vec::new());
    static OUT: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

fn with<R>(h: u32, f: impl FnOnce(&mut MultiSearch) -> R) -> Option<R> {
    SEARCHERS.with(|s| s.borrow_mut().get_mut(h as usize).and_then(|x| x.as_mut()).map(f))
}

fn put(bytes: &[u8]) {
    OUT.with(|o| {
        let mut o = o.borrow_mut();
        o.clear();
        o.extend_from_slice(bytes);
    });
}

fn put_f64(vals: &[f64]) {
    let mut bytes = Vec::with_capacity(vals.len() * 8);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    put(&bytes);
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 { &[] } else { std::slice::from_raw_parts(ptr, len) }
}

// ------------------------------------------------------------------ memory
#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(size.max(1));
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, size: usize) {
    drop(Vec::from_raw_parts(ptr, 0, size.max(1)));
}

#[no_mangle]
pub extern "C" fn out_ptr() -> *const u8 {
    OUT.with(|o| o.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn out_len() -> usize {
    OUT.with(|o| o.borrow().len())
}

// ---------------------------------------------------------------- searcher
#[no_mangle]
pub extern "C" fn searcher_new(n_trees: u32, c_puct: f32, fpu_reduction: f32, solver_empties: u32,
                               solver_budget: f64, cache_size: u32, seed: f64, variance_c_puct: u32,
                               q_init_weight: f32, draw_value: f32, forced_playouts: f32) -> u32 {
    let cfg = Config {
        c_puct, fpu_reduction, solver_empties, solver_budget: solver_budget as u64,
        cache_size: cache_size as usize, variance_c_puct: variance_c_puct != 0,
        q_init_weight, draw_value, forced_playouts,
    };
    let search = MultiSearch::new(n_trees.max(1) as usize, cfg, seed as u64);
    SEARCHERS.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(i) = s.iter().position(|x| x.is_none()) {
            s[i] = Some(search);
            i as u32
        } else {
            s.push(Some(search));
            (s.len() - 1) as u32
        }
    })
}

#[no_mangle]
pub extern "C" fn searcher_free(h: u32) {
    SEARCHERS.with(|s| {
        if let Some(slot) = s.borrow_mut().get_mut(h as usize) {
            *slot = None;
        }
    });
}

#[no_mangle]
pub extern "C" fn n_trees(h: u32) -> u32 {
    with(h, |m| m.trees.len() as u32).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn set_root_moves(h: u32, t: u32, ptr: *const u8, len: usize) -> i32 {
    let moves = slice(ptr, len).to_vec();
    with(h, |m| match Board::from_moves(&moves) {
        Some(b) => { m.trees[t as usize] = Tree::new(b); 0 }
        None => -1,
    }).unwrap_or(-2)
}

#[no_mangle]
pub unsafe extern "C" fn set_root_state(h: u32, t: u32, x: *const u16, o: *const u16,
                                        active: u32, stm: u32) -> i32 {
    let (x, o) = (slice(x, 9).to_vec(), slice(o, 9).to_vec());
    with(h, |m| match board_from_masks(&x, &o, active as u8, stm as u8) {
        Some(b) => { m.trees[t as usize] = Tree::new(b); 0 }
        None => -1,
    }).unwrap_or(-2)
}

#[no_mangle]
pub extern "C" fn advance(h: u32, t: u32, mv: u32) -> i32 {
    with(h, |m| if m.trees[t as usize].advance(mv as u8) { 0 } else { -1 }).unwrap_or(-2)
}

/// Leaves to evaluate: `n` canonical boards (81 bytes each) then `n` u32 tree ids.
#[no_mangle]
pub extern "C" fn collect(h: u32, max_leaves: u32, per_tree: u32, sims_limit: u32,
                          max_collisions: u32) -> u32 {
    with(h, |m| {
        let n = m.collect(max_leaves as usize, per_tree as usize, sims_limit, max_collisions as usize);
        let mut buf = vec![0u8; n * 81 + n * 4];
        for (i, p) in m.pending.iter().enumerate() {
            p.board.canonical(&mut buf[i * 81..i * 81 + 81]);
            buf[n * 81 + i * 4..n * 81 + i * 4 + 4].copy_from_slice(&p.tree.to_le_bytes());
        }
        put(&buf);
        n as u32
    }).unwrap_or(0)
}

/// Priors [n*81], values [n] and, when `q` is not null, Q [n*81], as f32.
#[no_mangle]
pub unsafe extern "C" fn apply(h: u32, priors: *const f32, values: *const f32, qs: *const f32) -> i32 {
    with(h, |m| {
        let n = m.pending.len();
        let p = slice(priors, n * 81).to_vec();
        let v = slice(values, n).to_vec();
        let q = if qs.is_null() { None } else { Some(slice(qs, n * 81).to_vec()) };
        m.apply(&p, &v, q.as_deref());
        0
    }).unwrap_or(-2)
}

#[no_mangle]
pub extern "C" fn pending_len(h: u32) -> u32 {
    with(h, |m| m.pending.len() as u32).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn set_limit(h: u32, t: u32, limit: u32) {
    with(h, |m| m.limits[t as usize] = limit);
}

#[no_mangle]
pub extern "C" fn root_expanded(h: u32, t: u32) -> u32 {
    with(h, |m| (m.trees[t as usize].root().state == 2) as u32).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn mix_root_noise(h: u32, t: u32, noise: *const f32, n: usize, eps: f32) -> u32 {
    let noise = slice(noise, n).to_vec();
    with(h, |m| m.trees[t as usize].mix_root_noise(&noise, eps) as u32).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn root_prior_temperature(h: u32, t: u32, temperature: f32) -> u32 {
    with(h, |m| m.trees[t as usize].root_prior_temperature(temperature) as u32).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn root_n(h: u32, t: u32) -> u32 {
    with(h, |m| m.trees[t as usize].root().n).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn root_value(h: u32, t: u32) -> f32 {
    with(h, |m| {
        let r = m.trees[t as usize].root();
        if r.solved != UNSOLVED {
            return solved_value(r.solved, m.cfg.draw_value);
        }
        if r.n == 0 { 0.0 } else { -(r.w / r.n as f64) as f32 }
    }).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn root_solved(h: u32, t: u32) -> i32 {
    with(h, |m| m.trees[t as usize].root().solved as i32).unwrap_or(UNSOLVED as i32)
}

/// (move, visits, q, prior, solved) per root child, five f64 each.
#[no_mangle]
pub extern "C" fn root_children(h: u32, t: u32) -> u32 {
    with(h, |m| {
        let tree = &m.trees[t as usize];
        if tree.root().state != 2 {
            put(&[]);
            return 0;
        }
        let kids = tree.children(0);
        let mut vals = Vec::with_capacity(kids.len() * 5);
        for c in kids {
            let q = if c.solved != UNSOLVED { -solved_value(c.solved, m.cfg.draw_value) }
                    else if c.n > 0 { (c.w / c.n as f64) as f32 } else { f32::NAN };
            let s = if c.solved == UNSOLVED { UNSOLVED } else { -c.solved };
            vals.extend_from_slice(&[c.mv as f64, c.n as f64, q as f64, c.prior as f64, s as f64]);
        }
        put_f64(&vals);
        kids.len() as u32
    }).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn root_canonical(h: u32, t: u32) -> u32 {
    with(h, |m| {
        let mut buf = [0u8; 81];
        m.trees[t as usize].root_board.canonical(&mut buf);
        put(&buf);
        81
    }).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn root_open_empties(h: u32, t: u32) -> u32 {
    with(h, |m| m.trees[t as usize].root_board.open_empties()).unwrap_or(0)
}

/// Exact value per root move: (move, value) as two signed bytes each.
#[no_mangle]
pub extern "C" fn root_solve(h: u32, t: u32, budget: f64, stop_on_win: u32, seconds: f64) -> u32 {
    with(h, |m| {
        let res = m.root_solve(t as usize, budget as u64, stop_on_win != 0, seconds);
        let mut buf = Vec::with_capacity(res.len() * 2);
        for (mv, v) in &res {
            buf.push(*mv);
            buf.push(*v as u8);
        }
        put(&buf);
        res.len() as u32
    }).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn pv(h: u32, t: u32, max_len: u32) -> u32 {
    with(h, |m| {
        let tree = &m.trees[t as usize];
        let mut out = Vec::new();
        let mut idx = 0usize;
        while out.len() < max_len as usize && tree.nodes[idx].state == 2 && tree.nodes[idx].n_children > 0 {
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
        put(&out);
        out.len() as u32
    }).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn node_count(h: u32, t: u32) -> u32 {
    with(h, |m| m.trees[t as usize].nodes.len() as u32).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn stats(h: u32) -> u32 {
    with(h, |m| {
        put_f64(&[m.cache_hits as f64, m.solver_calls as f64, m.solver_solved as f64]);
        3
    }).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn clear_cache(h: u32) {
    with(h, |m| m.clear_cache());
}

#[no_mangle]
pub extern "C" fn root_policy_target(h: u32, t: u32, k: f32) -> u32 {
    with(h, |m| {
        let target = m.trees[t as usize].root_policy_target(k);
        let vals: Vec<f64> = target.iter().flat_map(|(mv, p)| [*mv as f64, *p as f64]).collect();
        put_f64(&vals);
        target.len() as u32
    }).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn set_draw_value(h: u32, v: f32) { with(h, |m| m.cfg.draw_value = v); }
#[no_mangle]
pub extern "C" fn set_c_puct(h: u32, c: f32) { with(h, |m| m.cfg.c_puct = c); }
#[no_mangle]
pub extern "C" fn set_fpu_reduction(h: u32, f: f32) { with(h, |m| m.cfg.fpu_reduction = f); }
#[no_mangle]
pub extern "C" fn set_variance_c_puct(h: u32, on: u32) { with(h, |m| m.cfg.variance_c_puct = on != 0); }
#[no_mangle]
pub extern "C" fn set_q_init_weight(h: u32, w: f32) { with(h, |m| m.cfg.q_init_weight = w); }

// --------------------------------------------------------------- functions
/// Legal moves after `moves`: the moves, then `terminal`, then `winner`.  -1: illegal.
#[no_mangle]
pub unsafe extern "C" fn legal_moves(ptr: *const u8, len: usize) -> i32 {
    match Board::from_moves(slice(ptr, len)) {
        Some(b) => {
            let mut buf = [0u8; 81];
            let n = b.legal(&mut buf);
            let mut out = buf[..n].to_vec();
            out.push(b.is_terminal() as u8);
            out.push(b.winner);
            put(&out);
            n as i32
        }
        None => -1,
    }
}

/// Exact result for the side to move: -1/0/1, 2 over budget, -100 illegal.
#[no_mangle]
pub unsafe extern "C" fn solve(ptr: *const u8, len: usize, budget: f64) -> i32 {
    match Board::from_moves(slice(ptr, len)) {
        Some(b) => Solver::new().solve(&b, budget as u64).map(|v| v as i32).unwrap_or(2),
        None => -100,
    }
}

#[no_mangle]
pub unsafe extern "C" fn solve_state(x: *const u16, o: *const u16, active: u32, stm: u32, budget: f64) -> i32 {
    match board_from_masks(slice(x, 9), slice(o, 9), active as u8, stm as u8) {
        Some(b) => Solver::new().solve(&b, budget as u64).map(|v| v as i32).unwrap_or(2),
        None => -100,
    }
}

#[no_mangle]
pub unsafe extern "C" fn brute_force_state(x: *const u16, o: *const u16, active: u32, stm: u32) -> i32 {
    match board_from_masks(slice(x, 9), slice(o, 9), active as u8, stm as u8) {
        Some(b) => brute_force(&b) as i32,
        None => -100,
    }
}

#[no_mangle]
pub unsafe extern "C" fn canonical(ptr: *const u8, len: usize) -> i32 {
    match Board::from_moves(slice(ptr, len)) {
        Some(b) => {
            let mut buf = [0u8; 81];
            b.canonical(&mut buf);
            put(&buf);
            81
        }
        None => -1,
    }
}
