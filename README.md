# binary-tree
A simple rust binary-tree implementation.

It provides a Set like implementation similar to BTreeSet in the standard library, but is implemented with a binary search tree. 

You probably don't want to use this in production, but it's a fun little project to learn about binary trees and rust.

## Features
- Insertion
- Deletion
- Lookup
- Iterators
- Pretty printing
- Manual balancing with an implementation of [Day-Stout-Warren](https://en.wikipedia.org/wiki/Day%E2%80%93Stout%E2%80%93Warren_algorithm)


Example:
```rust
use binary_tree::BinaryTree;

fn main() {
    // Generate a random number between 10 and 100
    let random_count = rand::random::<u8>() % 90 + 10;

    // Insert into the binary tree, and count the number of successful insertions.
    // Duplicates are not inserted.
    let mut tree = BinaryTree::new();
    let mut inserted = 0;
    for _ in 0..random_count {
        if tree.insert(rand::random::<u8>()) {
            inserted += 1;
        }
    }
    println!("Generated {} random numbers and inserted {}.", random_count, inserted);
    tree.print();

    println!("\nBalancing the tree...");
    tree.balance();
    tree.print();
}
```
See [examples](examples) for more examples.
