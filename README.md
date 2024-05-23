# binary-tree Set

A simple set implementation in rust using a binary search tree.

It provides a partial implementation of the BTreeSet api in the standard library.

You probably don't want to use this in production, but it's a fun little project to learn about binary trees and rust.

## Example:

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

## Supported Methods

BTree method support is as follows:

| Method               | Supported |
|----------------------|----------|
| append               | ❌ |
| clear                | ✔️|
| contains             | ✔️|
| difference           | ❌|
| first                | ✔️️|
| get                  | ✔️|
| len                  | ✔️|
| insert               | ✔️|
| is_disjoint          | ❌|
| is_empty             | ✔️|
| is_subset            | ❌|
| is_superset          | ❌|
| iter                 | ✔️|
| last                 | ✔️|
| len                  | ✔️|
| new                  | ✔️|
| new_in               | ❌|
| pop_first            | ❌|
| pop_last             | ❌|
| range                | ❌|
| replace              | ❌|
| retain               | ❌|
| split_off            | ❌|
| symmetric_difference | ❌|
| take                 | ❌|
| union                | ❌|

## Supported Traits

BTree supports the following traits:

| Trait                     | Supported |
|---------------------------|-----------|
| BitAnd<&BinaryTree<T, A>> | ❌         |
| BitOr<&BinaryTree<T, A>>  | ❌         |
| BitXor<&BinaryTree<T, A>> | ❌         |
| Clone                     | ❌         |
| Debug                     | ❌         |
| Default                   | ✔️        |
| Eq                        | ✔️        |
| Extend<&'a T>             | ❌         |
| Extend<T>                 | ❌         |
| FromIterator<T>           | ✔️        |
| Hash                      | ❌         |
| IntoIterator (consuming)  | ✔️        |
| IntoIterator (reference)  | ✔️        |
| Ord                       | ❌         |
| PartialEq                 | ✔️        |
| PartialOrd                | ❌         |
| Sub<&BinaryTree<T, A>>    | ❌         |

## Additional Methods

- `balance`: manual balancing with an implementation
  of [Day-Stout-Warren](https://en.wikipedia.org/wiki/Day%E2%80%93Stout%E2%80%93Warren_algorithm)
- `depth`: the depth of the tree
- `print`: print the tree in a human-readable format
- `shape`: length, depth and whether it is route balanced
- `visit_in_order`: visit the tree in order
- `visit_in_reverse_order`: visit the tree in reverse order


