use binary_tree::BinaryTree;

fn main() {
    // Generate a random number between 10 and 100
    let random_count = rand::random::<u8>() % 90 + 10;

    // Insert into the binary tree, and count the number of successful insertions
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
