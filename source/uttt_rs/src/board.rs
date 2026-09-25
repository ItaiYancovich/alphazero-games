//! Ultimate Tic-Tac-Toe rules on bitboards.
//!
//! Identical to `alphazero_uttt/uttt_game.py`: cell = row * 9 + col; a won or
//! full small board is closed; the slot played sends the opponent to the board of
//! that index unless it is closed, in which case they may play anywhere; three
//! small boards in a line win, and every board closed with no line is a draw.

pub const ANY: u8 = 9;
pub const FULL: u16 = 0x1FF;

const LINES: [u16; 8] = [0b000000111, 0b000111000, 0b111000000, 0b001001001,
                         0b010010010, 0b100100100, 0b100010001, 0b001010100];

pub static WON: [bool; 512] = {
    let mut t = [false; 512];
    let mut m = 0;
    while m < 512 {
        let mut i = 0;
        while i < 8 {
            if (m as u16) & LINES[i] == LINES[i] {
                t[m] = true;
            }
            i += 1;
        }
        m += 1;
    }
    t
};

/// cell -> small board, cell -> slot, (board, slot) -> cell.
pub static BOARD_OF: [u8; 81] = {
    let mut t = [0u8; 81];
    let mut c = 0;
    while c < 81 {
        t[c] = ((c / 9 / 3) * 3 + (c % 9) / 3) as u8;
        c += 1;
    }
    t
};
pub static SLOT_OF: [u8; 81] = {
    let mut t = [0u8; 81];
    let mut c = 0;
    while c < 81 {
        t[c] = ((c / 9 % 3) * 3 + (c % 9) % 3) as u8;
        c += 1;
    }
    t
};
pub static CELL_AT: [[u8; 9]; 9] = {
    let mut t = [[0u8; 9]; 9];
    let mut b = 0;
    while b < 9 {
        let mut k = 0;
        while k < 9 {
            t[b][k] = (((b / 3) * 3 + k / 3) * 9 + (b % 3) * 3 + k % 3) as u8;
            k += 1;
        }
        b += 1;
    }
    t
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Board {
    pub small: [[u16; 9]; 2], // [player][board], player 0 = X, 1 = O
    pub meta: [u16; 2],
    pub closed: u16,
    pub active: u8, // forced board, or ANY
    pub stm: u8,    // side to move: 0 = X, 1 = O
    pub moves: u8,
    pub winner: u8, // 0 none, 1 X, 2 O
}

impl Default for Board {
    fn default() -> Self {
        Board { small: [[0; 9]; 2], meta: [0; 2], closed: 0, active: ANY, stm: 0, moves: 0, winner: 0 }
    }
}

impl Board {
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.winner != 0 || self.closed == FULL
    }

    /// Value of a finished position for the side to move: -1 lost, 0 drawn.
    #[inline]
    pub fn terminal_value(&self) -> f32 {
        if self.winner != 0 { -1.0 } else { 0.0 }
    }

    #[inline]
    pub fn occupied(&self, b: usize) -> u16 {
        self.small[0][b] | self.small[1][b]
    }

    /// Legal cells into `out`, ascending; returns the count.
    pub fn legal(&self, out: &mut [u8; 81]) -> usize {
        if self.is_terminal() {
            return 0;
        }
        let mut n = 0;
        let mut push_board = |b: usize, n: &mut usize| {
            let mut free = !self.occupied(b) & FULL;
            while free != 0 {
                let k = free.trailing_zeros() as usize;
                out[*n] = CELL_AT[b][k];
                *n += 1;
                free &= free - 1;
            }
        };
        if self.active != ANY {
            push_board(self.active as usize, &mut n);
        } else {
            for b in 0..9 {
                if (self.closed >> b) & 1 == 0 {
                    push_board(b, &mut n);
                }
            }
        }
        out[..n].sort_unstable();
        n
    }

    /// Bitmask of legal cells (bit c = cell c).
    pub fn legal_mask(&self) -> u128 {
        let mut buf = [0u8; 81];
        let n = self.legal(&mut buf);
        let mut m = 0u128;
        for &c in &buf[..n] {
            m |= 1u128 << c;
        }
        m
    }

    pub fn is_legal(&self, cell: u8) -> bool {
        if self.is_terminal() || cell >= 81 {
            return false;
        }
        let b = BOARD_OF[cell as usize] as usize;
        if (self.closed >> b) & 1 == 1 {
            return false;
        }
        if self.active != ANY && b != self.active as usize {
            return false;
        }
        (self.occupied(b) >> SLOT_OF[cell as usize]) & 1 == 0
    }

    #[inline]
    pub fn play(&mut self, cell: u8) {
        let b = BOARD_OF[cell as usize] as usize;
        let k = SLOT_OF[cell as usize] as usize;
        let p = self.stm as usize;
        let mine = self.small[p][b] | (1 << k);
        self.small[p][b] = mine;
        self.moves += 1;
        if WON[mine as usize] {
            self.meta[p] |= 1 << b;
            self.closed |= 1 << b;
            if WON[self.meta[p] as usize] {
                self.winner = (p + 1) as u8;
            }
        } else if mine | self.small[1 - p][b] == FULL {
            self.closed |= 1 << b;
        }
        self.stm ^= 1;
        self.active = if (self.closed >> k) & 1 == 1 { ANY } else { k as u8 };
    }

    /// Would the side to move win the whole game by playing `cell`?
    #[inline]
    pub fn wins_game(&self, cell: u8) -> bool {
        let b = BOARD_OF[cell as usize] as usize;
        let k = SLOT_OF[cell as usize] as usize;
        let p = self.stm as usize;
        WON[(self.small[p][b] | (1 << k)) as usize] && WON[(self.meta[p] | (1 << b)) as usize]
    }

    /// Canonical 81-cell view: 0 playable, 1 own, 2 opponent, 3 empty but unreachable.
    pub fn canonical(&self, out: &mut [u8]) {
        let me = self.stm as usize;
        for c in 0..81 {
            out[c] = 3;
        }
        for b in 0..9 {
            for k in 0..9 {
                let c = CELL_AT[b][k] as usize;
                if (self.small[me][b] >> k) & 1 == 1 {
                    out[c] = 1;
                } else if (self.small[1 - me][b] >> k) & 1 == 1 {
                    out[c] = 2;
                }
            }
        }
        let mut buf = [0u8; 81];
        let n = self.legal(&mut buf);
        for &c in &buf[..n] {
            out[c as usize] = 0;
        }
    }

    /// Empty cells in boards that are still open.
    pub fn open_empties(&self) -> u32 {
        let mut n = 0;
        for b in 0..9 {
            if (self.closed >> b) & 1 == 0 {
                n += 9 - self.occupied(b).count_ones();
            }
        }
        n
    }

    /// 64-bit position key (marks, forced board, side to move).
    pub fn key(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for p in 0..2 {
            for b in 0..9 {
                h ^= self.small[p][b] as u64;
                h = h.wrapping_mul(0x100000001b3);
                h ^= h >> 29;
            }
        }
        h ^= (self.active as u64) << 1 | self.stm as u64;
        h = h.wrapping_mul(0x9E3779B97F4A7C15);
        h ^ (h >> 32)
    }

    pub fn from_moves(moves: &[u8]) -> Option<Board> {
        let mut b = Board::default();
        for &m in moves {
            if !b.is_legal(m) {
                return None;
            }
            b.play(m);
        }
        Some(b)
    }
}

/// A tiny xorshift RNG, so the crate needs no dependencies.
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 11) % n as u64) as usize
    }
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}
