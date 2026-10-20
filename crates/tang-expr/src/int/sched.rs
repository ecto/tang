//! Scheduling shared by every backend: live nodes, scopes, loop placement,
//! and which nodes may be inlined into their single user.

use super::{IId, INode, IntGraph};

/// One step of a scope's body.
pub(crate) enum Item {
    /// Compute a node.
    Node(IId),
    /// Run loop `l`: `pre` computes what the condition needs, then the
    /// condition is checked, then `post` and the state update.
    Loop {
        l: u32,
        pre: Vec<Item>,
        post: Vec<Item>,
    },
}

pub(crate) struct Sched {
    pub items: Vec<Item>,
    /// Inline this node into its only user (same scope, one use).
    pub inline: Vec<bool>,
}

impl IntGraph {
    pub(crate) fn operands(&self, n: INode, mut f: impl FnMut(IId)) {
        match n {
            INode::Bin(_, a, b) | INode::Cmp(_, a, b) => {
                f(a);
                f(b);
            }
            INode::Un(_, a) | INode::Bitcast(_, a) | INode::Load(_, a) => f(a),
            INode::LoadRw(_, i, t) => {
                f(i);
                f(t);
            }
            INode::Store(_, i, v, t) => {
                f(i);
                f(v);
                f(t);
            }
            INode::Select(c, a, b) => {
                f(c);
                f(a);
                f(b);
            }
            INode::Const(..)
            | INode::Param(..)
            | INode::State(..)
            | INode::LoopOut(..)
            | INode::Tok => {}
        }
    }

    pub(crate) fn schedule(&self, outputs: &[IId]) -> Sched {
        let n = self.nodes.len();
        let mut live = vec![false; n];
        let mut loop_live = vec![false; self.loops.len()];
        let mut uses = vec![0u32; n];
        let mut user_scope: Vec<Option<Option<u32>>> = vec![None; n];
        let mut stack: Vec<IId> = Vec::new();
        let use_at = |x: IId,
                      s: Option<u32>,
                      uses: &mut Vec<u32>,
                      us: &mut Vec<Option<Option<u32>>>,
                      st: &mut Vec<IId>| {
            uses[x.index()] += 1;
            us[x.index()] = Some(s);
            st.push(x);
        };
        for &o in outputs {
            assert_eq!(
                self.scope[o.index()],
                None,
                "int graph: output inside a loop"
            );
            use_at(o, None, &mut uses, &mut user_scope, &mut stack);
        }
        while let Some(x) = stack.pop() {
            if live[x.index()] {
                continue;
            }
            live[x.index()] = true;
            let node = self.nodes[x.index()];
            let s = self.scope[x.index()];
            let mut ops = Vec::new();
            self.operands(node, |a| ops.push(a));
            for a in ops {
                use_at(a, s, &mut uses, &mut user_scope, &mut stack);
            }
            if let INode::LoopOut(l, _) | INode::State(l, _) = node {
                if !loop_live[l as usize] {
                    loop_live[l as usize] = true;
                    let lp = &self.loops[l as usize];
                    for &i in &lp.init {
                        use_at(i, lp.parent, &mut uses, &mut user_scope, &mut stack);
                    }
                    use_at(lp.cond, Some(l), &mut uses, &mut user_scope, &mut stack);
                    for &x in &lp.next {
                        use_at(x, Some(l), &mut uses, &mut user_scope, &mut stack);
                    }
                }
            }
        }
        let mut inline: Vec<bool> = (0..n)
            .map(|i| {
                // effects stay at their place in the schedule
                let leaf = matches!(
                    self.nodes[i],
                    INode::State(..) | INode::LoopOut(..) | INode::LoadRw(..) | INode::Store(..)
                );
                live[i] && !leaf && uses[i] == 1 && user_scope[i] == Some(self.scope[i])
            })
            .collect();
        // bound the nesting of inlined expressions (lowerings recurse on it)
        let mut depth = vec![0u32; n];
        for i in 0..n {
            let mut d = 0;
            self.operands(self.nodes[i], |a| {
                if inline[a.index()] {
                    d = d.max(depth[a.index()]);
                }
            });
            depth[i] = d + 1;
            if depth[i] >= 32 {
                inline[i] = false;
                depth[i] = 0;
            }
        }
        let mut done = vec![false; self.loops.len()];
        let items = self.scope_items(None, &live, None, &mut done);
        Sched { items, inline }
    }

    /// Items of `scope` in id order (a loop goes right before its first
    /// `LoopOut`), restricted to `only` when given.
    fn scope_items(
        &self,
        scope: Option<u32>,
        live: &[bool],
        only: Option<&[bool]>,
        done: &mut Vec<bool>,
    ) -> Vec<Item> {
        let mut items = Vec::new();
        for i in 0..self.nodes.len() {
            if !live[i] || self.scope[i] != scope || only.is_some_and(|m| !m[i]) {
                continue;
            }
            match self.nodes[i] {
                INode::State(..) => continue,
                INode::LoopOut(l, _) if !done[l as usize] => {
                    done[l as usize] = true;
                    items.push(self.loop_item(l, live, done));
                }
                _ => {}
            }
            items.push(Item::Node(IId(i as u32)));
        }
        items
    }

    fn loop_item(&self, l: u32, live: &[bool], done: &mut Vec<bool>) -> Item {
        // nodes of this loop's scope that the condition needs
        let mut mark = vec![false; self.nodes.len()];
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![self.loops[l as usize].cond];
        while let Some(x) = stack.pop() {
            if seen[x.index()] {
                continue;
            }
            seen[x.index()] = true;
            let s = self.scope[x.index()];
            if s == Some(l) {
                mark[x.index()] = true;
            } else if self.depth(s) <= self.depth(Some(l)) {
                continue; // enclosing scope: already computed
            }
            let node = self.nodes[x.index()];
            self.operands(node, |a| stack.push(a));
            if let INode::LoopOut(m, _) = node {
                let lp = &self.loops[m as usize];
                stack.extend(lp.init.iter().copied());
                stack.push(lp.cond);
                stack.extend(lp.next.iter().copied());
            }
        }
        let pre = self.scope_items(Some(l), live, Some(&mark), done);
        let rest: Vec<bool> = mark.iter().map(|&m| !m).collect();
        let post = self.scope_items(Some(l), live, Some(&rest), done);
        Item::Loop { l, pre, post }
    }
}
