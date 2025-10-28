use crate::ripser::types::{IndexT, ValueT};

// Union-Find data structure
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<IndexT>,
    rank: Vec<u8>,
    birth: Vec<ValueT>,
}

impl UnionFind {
    pub fn new(n: IndexT) -> Self {
        let n_usize = n as usize;
        let mut parent = vec![0; n_usize];
        for (i, item) in parent.iter_mut().enumerate().take(n_usize) {
            *item = i as IndexT;
        }

        Self {
            parent,
            rank: vec![0; n_usize],
            birth: vec![0.0; n_usize],
        }
    }

    pub fn set_birth(&mut self, i: IndexT, val: ValueT) {
        self.birth[i as usize] = val;
    }

    pub fn get_birth(&self, i: IndexT) -> ValueT {
        self.birth[i as usize]
    }

    pub fn find(&mut self, mut x: IndexT) -> IndexT {
        let mut y = x;
        let mut z = self.parent[y as usize];
        while z != y {
            y = z;
            z = self.parent[y as usize];
        }
        let root = z;
        y = self.parent[x as usize];
        while root != y {
            self.parent[x as usize] = root;
            x = y;
            y = self.parent[x as usize];
        }
        root
    }

    pub fn link(&mut self, x: IndexT, y: IndexT) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        if x_root == y_root {
            return;
        }

        let x_rank = self.rank[x_root as usize];
        let y_rank = self.rank[y_root as usize];

        if x_rank > y_rank {
            self.parent[y_root as usize] = x_root;
            self.birth[x_root as usize] =
                self.birth[x_root as usize].min(self.birth[y_root as usize]);
        } else {
            self.parent[x_root as usize] = y_root;
            self.birth[y_root as usize] =
                self.birth[x_root as usize].min(self.birth[y_root as usize]);
            if x_rank == y_rank {
                self.rank[y_root as usize] += 1;
            }
        }
    }
}
