use std::cmp::max;
use std::collections::HashMap;
use std::fmt::Display;

pub struct TreeNode<T: Ord> {
    value: T,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
}

pub struct BinaryTree<T: Ord> {
    root: Option<Box<TreeNode<T>>>,
}

impl<T: Ord> BinaryTree<T> {
    pub fn new() -> Self {
        BinaryTree { root: None }
    }

    /// Balance the binary tree using the [Day-Stout-Warren algorithm](https://en.wikipedia.org/wiki/Day%E2%80%93Stout%E2%80%93Warren_algorithm)
    ///
    /// The algorithm first converts the binary tree into a vine (a degenerate tree) by rotating nodes in place.
    /// It then compresses the vine by rotating nodes in place to form a balanced tree.
    ///
    pub fn balance(&mut self) {
        let mut size = self.tree_to_vine();
        let power = ((size + 1) as f64).log2().floor() as usize;
        let leaf_count = size + 1 - (2usize).pow(power as u32);
        self.compress(leaf_count);
        size -= leaf_count;
        while size > 1 {
            self.compress(size / 2);
            size /= 2;
        }
    }

    pub fn contains<Q>(&self, value: &Q) -> bool
        where
            T: std::borrow::Borrow<Q>,
            Q: Ord,
    {
        self.get(value).is_some()
    }

    pub fn depth(&self) -> usize {
        let mut depth: usize = 0;
        self.visit_in_order(|_, level| {
            depth = max(level, depth);
        });
        depth
    }

    pub fn get<Q>(&self, value: &Q) -> Option<&T>
        where
            T: std::borrow::Borrow<Q>,
            Q: Ord,
    {
        let mut current = self.root.as_deref();
        while let Some(node) = current {
            match value.cmp(node.value.borrow()) {
                std::cmp::Ordering::Less => current = node.left.as_deref(),
                std::cmp::Ordering::Greater => current = node.right.as_deref(),
                std::cmp::Ordering::Equal => return Some(&node.value),
            }
        }
        None
    }

    pub fn insert(&mut self, value: T) -> bool {
        let node = Box::new(TreeNode {
            value,
            left: None,
            right: None,
        });
        let (next_root, inserted) = Self::insert_node(self.root.take(), node);
        self.root = Some(next_root);
        inserted
    }

    pub fn iter(&self) -> TreeIter<T> {
        let mut iter = TreeIter { nodes: Vec::new() };
        iter.push_left(self.root.as_deref());
        iter
    }

    pub fn len(&self) -> usize {
        self.iter().count()
    }

    pub fn remove(&mut self, value: T) -> bool {
        let (next_root, removed) = Self::remove_node(self.root.take(), &value);
        self.root = next_root;
        removed
    }

    /// Return the size, depth, and whether the binary tree is route balanced.
    ///
    /// A binary tree is route balanced if the depth of the tree is less than or equal to the
    /// maximum depth of a binary tree with the same number of nodes and every level except the
    /// last is completely filled.
    pub fn shape(&self) -> (usize, usize, bool) {
        let mut depth: usize = 0;
        let mut size: usize = 0;
        let mut depth_count: HashMap<usize, usize> = HashMap::new();
        self.visit_in_order(|_, level| {
            depth = max(level, depth);
            size += 1;
            *depth_count.entry(level).or_insert(0) += 1;
        });
        let max_depth = ((size + 1) as f64).log2().floor() as usize;
        let route_balanced = depth <= max_depth
            && depth_count
            .into_iter()
            .filter(|(depth, _)| *depth < max_depth)
            .all(|(depth, count)| count == 2usize.pow(depth as u32));
        (size, depth, route_balanced)
    }

    pub fn visit_in_order<F>(&self, mut f: F)
        where
            F: FnMut(&Box<TreeNode<T>>, usize),
    {
        Self::in_order_traversal(&self.root, &mut f, 0);
    }

    pub fn visit_in_reverse_order<F>(&self, mut f: F)
        where
            F: FnMut(&Box<TreeNode<T>>, usize),
    {
        Self::reverse_order_traversal(&self.root, &mut f, 0);
    }

    fn compress(&mut self, count: usize) {
        let mut current = self.root.as_deref_mut();
        for _ in 0..count {
            if let Some(node) = current.take() {
                let right_node = node.right.take();
                if right_node.is_some() {
                    // Rotate nodes in place by swapping values and updating left and right pointers.
                    let mut right_node = right_node.unwrap();
                    std::mem::swap(&mut node.value, &mut right_node.value);
                    node.right = right_node.right;
                    right_node.right = right_node.left.take();
                    right_node.left = node.left.take();
                    node.left = Some(right_node);
                    current = node.right.as_deref_mut();
                }
            }
        }
    }

    fn in_order_traversal<F>(node: &Option<Box<TreeNode<T>>>, f: &mut F, level: usize)
        where
            F: FnMut(&Box<TreeNode<T>>, usize),
    {
        if let Some(ref boxed_node) = node {
            Self::in_order_traversal(&boxed_node.left, f, level + 1);
            f(boxed_node, level);
            Self::in_order_traversal(&boxed_node.right, f, level + 1);
        }
    }

    fn insert_node(
        node: Option<Box<TreeNode<T>>>,
        new_node: Box<TreeNode<T>>,
    ) -> (Box<TreeNode<T>>, bool) {
        match node {
            Some(mut old_node) => match new_node.value.cmp(&old_node.value) {
                std::cmp::Ordering::Less => {
                    let (new_left, inserted) = Self::insert_node(old_node.left.take(), new_node);
                    old_node.left = Some(new_left);
                    (old_node, inserted)
                }
                std::cmp::Ordering::Greater => {
                    let (new_right, inserted) = Self::insert_node(old_node.right.take(), new_node);
                    old_node.right = Some(new_right);
                    (old_node, inserted)
                }
                std::cmp::Ordering::Equal => (old_node, false),
            },
            None => (new_node, true),
        }
    }

    fn remove_node(node: Option<Box<TreeNode<T>>>, value: &T) -> (Option<Box<TreeNode<T>>>, bool) {
        match node {
            Some(mut boxed_node) => match value.cmp(&boxed_node.value) {
                std::cmp::Ordering::Less => {
                    let (new_left, removed) = Self::remove_node(boxed_node.left.take(), value);
                    boxed_node.left = new_left;
                    (Some(boxed_node), removed)
                }
                std::cmp::Ordering::Greater => {
                    let (new_right, removed) = Self::remove_node(boxed_node.right.take(), value);
                    boxed_node.right = new_right;
                    (Some(boxed_node), removed)
                }
                std::cmp::Ordering::Equal => {
                    match (boxed_node.left.take(), boxed_node.right.take()) {
                        (None, None) => (None, true),
                        (Some(left), None) => (Some(left), true),
                        (None, Some(right)) => (Some(right), true),
                        (Some(left), Some(right)) => {
                            let (new_left, right_most_value) = Self::remove_right_most_node(left);
                            boxed_node.left = new_left;
                            boxed_node.value = right_most_value;
                            boxed_node.right = Some(right);
                            (Some(boxed_node), true)
                        }
                    }
                }
            },
            None => (None, false),
        }
    }

    fn remove_right_most_node(mut node: Box<TreeNode<T>>) -> (Option<Box<TreeNode<T>>>, T) {
        match node.right {
            Some(right_node) => {
                let (new_right, right_most_value) = Self::remove_right_most_node(right_node);
                node.right = new_right;
                (Some(node), right_most_value)
            }
            None => (node.left, node.value),
        }
    }

    fn reverse_order_traversal<F>(node: &Option<Box<TreeNode<T>>>, f: &mut F, level: usize)
        where
            F: FnMut(&Box<TreeNode<T>>, usize),
    {
        if let Some(ref boxed_node) = node {
            Self::reverse_order_traversal(&boxed_node.right, f, level + 1);
            f(boxed_node, level);
            Self::reverse_order_traversal(&boxed_node.left, f, level + 1);
        }
    }

    fn tree_to_vine(&mut self) -> usize {
        let mut possible_remainder = self.root.as_deref_mut();
        let mut size = 0;

        while let Some(remainder) = possible_remainder.take() {
            if remainder.left.is_none() {
                possible_remainder = remainder.right.as_deref_mut();
                size += 1;
            } else {
                // Rotate nodes in place by swapping values and updating left and right pointers.
                let mut left_node = remainder.left.take().unwrap();
                std::mem::swap(&mut remainder.value, &mut left_node.value);
                remainder.left = left_node.left.take();
                left_node.left = left_node.right.take();
                left_node.right = remainder.right.take();
                remainder.right = Some(left_node);
                possible_remainder = Some(remainder);
            }
        }
        size
    }
}

impl<T: Ord + Display> BinaryTree<T> {
    pub fn print(&self) {
        self.visit_in_reverse_order(|node, level| {
            for _ in 0..level {
                print!("\t");
            }
            println!("{}", node.value);
        });
        let (size, depth, route_balanced) = self.shape();
        println!(
            "Size: {}, Depth: {}, Route Balanced: {}\n",
            size, depth, route_balanced
        );
    }
}

impl<T: Ord> Default for BinaryTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord> Eq for BinaryTree<T> {}

impl<T: Ord, const N: usize> From<[T; N]> for BinaryTree<T> {
    fn from(array: [T; N]) -> Self {
        Self::from_iter(array)
    }
}

impl<T: Ord> FromIterator<T> for BinaryTree<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let mut tree = BinaryTree::new();
        for value in iter {
            tree.insert(value);
        }
        tree
    }
}

impl<'a, T: Ord> IntoIterator for &'a BinaryTree<T> {
    type Item = &'a T;
    type IntoIter = TreeIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        BinaryTree::iter(self)
    }
}

pub struct TreeIter<'a, T: Ord> {
    nodes: Vec<&'a TreeNode<T>>,
}

impl<'a, T: Ord> TreeIter<'a, T> {
    fn push_left(&mut self, mut possible_next_node: Option<&'a TreeNode<T>>) {
        while let Some(node) = possible_next_node {
            self.nodes.push(node);
            possible_next_node = node.left.as_deref();
        }
    }
}

impl<'a, T: Ord> Iterator for TreeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.nodes.pop()?;
        self.push_left(node.right.as_deref());
        Some(&node.value)
    }
}

impl<T: Ord> IntoIterator for BinaryTree<T> {
    type Item = T;
    type IntoIter = TreeIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let mut iter = TreeIntoIter { nodes: Vec::new() };
        iter.push_left(self.root);
        iter
    }
}

pub struct TreeIntoIter<T: Ord> {
    nodes: Vec<TreeNode<T>>,
}

impl<T: Ord> TreeIntoIter<T> {
    fn push_left(&mut self, mut possible_next_node: Option<Box<TreeNode<T>>>) {
        while let Some(mut node) = possible_next_node {
            possible_next_node = node.left.take();
            if let Some(right) = node.right.take() {
                self.nodes.push(*right);
            }
            self.nodes.push(*node);
        }
    }
}

impl<T: Ord> Iterator for TreeIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.nodes.pop()?;
        self.push_left(node.right);
        Some(node.value)
    }
}

impl<T: Ord> PartialEq<Self> for BinaryTree<T> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains() {
        let tree = BinaryTree::from_iter(vec![5, 3, 7, 1, 4, 6, 8]);
        assert!(tree.contains(&3));
        assert!(tree.contains(&7));
        assert!(!tree.contains(&10));
    }

    #[test]
    fn test_get() {
        let tree = BinaryTree::from_iter(vec![5, 3, 7, 1, 4, 6, 8]);
        assert_eq!(tree.get(&3), Some(&3));
        assert_eq!(tree.get(&7), Some(&7));
        assert_eq!(tree.get(&10), None);
    }

    #[test]
    fn test_len() {
        let tree = BinaryTree::from_iter(vec![5, 3, 7, 1, 4, 6, 8]);
        assert_eq!(tree.len(), 7);
        let tree: BinaryTree<u8> = BinaryTree::new();
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_eq() {
        let tree1 = BinaryTree::from_iter(vec![5, 3, 7, 1, 4, 6, 8]);
        let tree2 = BinaryTree::from_iter(vec![7, 3, 5, 1, 4, 6, 8]);
        assert!(tree1.eq(&tree2));

        let tree3 = BinaryTree::from_iter(vec![5, 3, 7, 1, 4, 6]);
        assert!(!tree1.eq(&tree3));

        let empty_tree_one: BinaryTree<u8> = BinaryTree::new();
        let empty_tree_two: BinaryTree<u8> = BinaryTree::new();
        assert!(empty_tree_one.eq(&empty_tree_two));
    }

    #[test]
    fn test_insert() {
        let mut tree = BinaryTree::<i32>::new();
        let test_values = [5, 3, 7, 1, 4, 6, 8, 1, 3];
        let results = test_values
            .iter()
            .map(|&v| tree.insert(v))
            .collect::<Vec<_>>();
        assert_eq!(
            results,
            vec![true, true, true, true, true, true, true, false, false]
        );
        assert!(tree.iter().cloned().eq([1, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn test_insert_from_iter_array() {
        let tree = BinaryTree::from_iter([5, 3, 7, 1, 4, 6, 8]);
        assert!(tree.iter().cloned().eq([1, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn test_insert_from_array() {
        let tree = BinaryTree::from([5, 3, 7, 1, 4, 6, 8]);
        assert!(tree.iter().cloned().eq([1, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn test_iterable_ref() {
        let tree = BinaryTree::from_iter(vec![5, 3, 7]);
        let mut results: Vec<&i32> = Vec::new();
        for node in &tree {
            results.push(node);
        }
        assert_eq!(results, vec![&3, &5, &7]);
        assert!(tree.iter().cloned().eq(vec![3, 5, 7]));
    }

    #[test]
    fn test_consuming_iterator() {
        let tree = BinaryTree::from_iter(vec![5, 3, 7]);
        let mut results: Vec<i32> = Vec::new();
        for node in tree {
            results.push(node);
        }
        assert_eq!(results, vec![3, 5, 7]);
    }

    #[test]
    fn test_remove() {
        let mut tree = BinaryTree::from_iter(vec![5, 3, 7, 1, 4, 6, 8]);
        assert!(tree.iter().cloned().eq(vec![1, 3, 4, 5, 6, 7, 8]));

        // Remove a value and check if it's deleted correctly
        assert!(tree.remove(3));
        assert!(tree.iter().cloned().eq(vec![1, 4, 5, 6, 7, 8]));

        // Remove another value and check if it's deleted correctly
        assert!(tree.remove(7));
        assert!(tree.iter().cloned().eq(vec![1, 4, 5, 6, 8]));

        // Remove a value that doesn't exist and check if the tree remains the same
        assert!(!tree.remove(10));
        assert!(tree.iter().cloned().eq(vec![1, 4, 5, 6, 8]));
    }

    #[test]
    fn test_remove_all() {
        let mut initial_values = vec![5, 3, 7, 1, 4, 6, 8];
        let mut tree = BinaryTree::from_iter(initial_values.clone());
        initial_values.sort_unstable();

        for value in initial_values.clone() {
            assert!(tree.remove(value));
            initial_values.retain(|&v| v != value);
            assert!(&tree.iter().eq(&initial_values));
        }

        // At this point, all values should have been deleted from the tree
        assert!(tree.iter().next().is_none());
    }

    #[test]
    fn test_balance_ordered_tree() {
        let mut tree = BinaryTree::from_iter(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        assert_eq!((13, 12, false), tree.shape());
        tree.balance();
        assert!(
            &tree.iter().cloned().eq(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]));
        assert_eq!((13, 3, true), tree.shape());
        assert_eq!(tree.depth(), 3);
    }

    #[test]
    fn test_balance_reverse_ordered_tree() {
        let mut tree: BinaryTree<String> = BinaryTree::from_iter(vec![
            "Grape".into(),
            "Fig".into(),
            "Elderberry".into(),
            "Date".into(),
            "Cherry".into(),
            "Banana".into(),
            "Apple".into(),
        ]);
        assert_eq!((7, 6, false), tree.shape());
        tree.balance();
        assert!(
            tree.iter().cloned().eq(vec![
                "Apple",
                "Banana",
                "Cherry",
                "Date",
                "Elderberry",
                "Fig",
                "Grape",
            ])
        );
        assert_eq!((7, 2, true), tree.shape());
        assert_eq!(tree.depth(), 2);
    }
}
