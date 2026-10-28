//! Residual slots shared by IR dims whose lifetimes do not overlap: qwenputer's
//! `q35/emit.py` `allocate` (emit.py:80; MIT, Copyright (c) 2026 Divyansh Agrawal).
//!
//! A residual write adds, so a slot is handed over by subtracting its old value: an MLP node
//! defining the new dim does it with one extra unit (`subtract`); a head's dim is cleared by a
//! unit in the MLP sublayer just before it (`clears`).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::{Graph, Node, ONE};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Alloc {
    /// IR dim → residual slot.
    pub slots: HashMap<u32, usize>,
    pub n_slots: usize,
    /// FFN-defined dim → the old dim its MLP subtracts.
    pub subtract: HashMap<u32, u32>,
    /// MLP level → old dims cleared there.
    pub clears: BTreeMap<u32, Vec<u32>>,
    /// Each slot's last owner.
    pub final_: Vec<u32>,
}

/// emit.py:80. `keep` dims are never handed over; `unit_cap` is the MLP width. (`clearable` is
/// python's optional restriction, unused by the store build: `None` there.)
pub fn allocate(g: &Graph, keep: &[u32], unit_cap: usize) -> Alloc {
    let top = g.nodes.iter().map(|n| n.level()).max().unwrap_or(0) + 1;
    // emit.py:87-93: each dim's last reader
    let mut last: Vec<u32> = g.dims.iter().map(|d| d.level).collect();
    for n in &g.nodes {
        for d in n.reads() {
            if d != ONE {
                last[d as usize] = last[d as usize].max(n.level());
            }
        }
    }
    let keep: HashSet<u32> = keep.iter().copied().collect();
    for &d in &keep {
        last[d as usize] = top;
    }
    let mut units: HashMap<u32, usize> = g.units.iter().map(|(&k, &v)| (k, v)).collect();
    // emit.py:98: a head whose outputs may be tiny keeps its slots
    let sticky: HashSet<u32> = g
        .nodes
        .iter()
        .filter_map(|n| match n {
            Node::Head(h) if h.tiny > 0.0 && h.tiny < 2f64.powi(-100) => Some(h),
            _ => None,
        })
        .flat_map(|h| h.vheads.iter().flat_map(|vh| vh.out.iter().map(|o| o.1)))
        .collect();
    let big: HashSet<u32> = g.big.iter().copied().collect();
    let mut a = Alloc::default();
    let mut owner: Vec<u32> = vec![];

    // emit.py:105 `take`
    let mut take = |a: &mut Alloc, owner: &mut Vec<u32>, d: u32, level: u32, by_ffn: bool| {
        for (s, &o) in owner.iter().enumerate() {
            if keep.contains(&o) || sticky.contains(&o) {
                continue;
            }
            let born = g.dims[o as usize].level;
            let lo = last[o as usize];
            if by_ffn
                && !big.contains(&o)
                && lo <= level
                && born < level
                && *units.get(&level).unwrap_or(&0) < unit_cap
            {
                *units.entry(level).or_default() += 1;
                a.subtract.insert(d, o);
            } else if !by_ffn
                && level >= 3
                && lo < level
                && born < level - 1
                && *units.get(&(level - 1)).unwrap_or(&0) < unit_cap
            {
                *units.entry(level - 1).or_default() += 1;
                a.clears.entry(level - 1).or_default().push(o);
            } else {
                continue;
            }
            owner[s] = d;
            a.slots.insert(d, s);
            return;
        }
        a.slots.insert(d, owner.len());
        owner.push(d);
    };

    // emit.py:126-138
    for &d in &g.token_dims {
        take(&mut a, &mut owner, d, 0, false);
    }
    let mut defined: BTreeMap<u32, Vec<(u32, bool)>> = BTreeMap::new();
    for n in &g.nodes {
        match n {
            Node::Ffn(f) if !f.into => defined.entry(f.level).or_default().push((f.out, true)),
            Node::Ffn(_) => {}
            Node::Head(h) => {
                for vh in h.vheads.iter().filter(|vh| !vh.into) {
                    defined
                        .entry(h.level)
                        .or_default()
                        .extend(vh.out.iter().map(|o| (o.1, false)));
                }
            }
        }
    }
    for (level, ds) in defined {
        for (d, by_ffn) in ds {
            take(&mut a, &mut owner, d, level, by_ffn);
        }
    }
    a.n_slots = owner.len();
    a.final_ = owner;
    a
}
