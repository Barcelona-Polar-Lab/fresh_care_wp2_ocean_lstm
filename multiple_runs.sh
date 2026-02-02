# Define hyperparameter combinations: [batch_size, dropout_rate, learning_rate]
hyperparams=(
    # Batch size = 8
    "8 0.1 0.0001"
    "8 0.3 0.0001"
    "8 0.15 0.0005"
    "8 0.4 0.0005"
    "8 0.1 0.001"
    "8 0.2 0.001"
    "8 0.15 0.005"
    "8 0.4 0.005"

    # Batch size = 16
    "16 0.15 0.0001"
    "16 0.3 0.0001"
    "16 0.4 0.0001"
    "16 0.1 0.0005"
    "16 0.2 0.0005"
    "16 0.15 0.001"
    "16 0.3 0.001"
    "16 0.4 0.001"
    "16 0.15 0.005"
    "16 0.3 0.005"

    # Batch size = 32
    "32 0.1 0.0001"
    "32 0.15 0.0001"
    "32 0.2 0.0001"
    "32 0.4 0.0001"
    "32 0.15 0.0005"
    "32 0.3 0.0005"
    "32 0.15 0.001"
    "32 0.2 0.001"
    "32 0.4 0.001"
    "32 0.1 0.005"
    "32 0.3 0.005"
)

# Counter for tracking progress
total=${#hyperparams[@]}
current=0

for params in "${hyperparams[@]}"; do
    current=$((current + 1))
    read -r batch_size dropout_rate learning_rate <<< "$params"
    
    echo "[$current/$total] Running with: Batch=$batch_size, Dropout=$dropout_rate, LR=$learning_rate"
    python3 lstm_pytorch.py --batch_size $batch_size --dropout_rate $dropout_rate --learning_rate $learning_rate
    echo "Completed: Batch=$batch_size, Dropout=$dropout_rate, LR=$learning_rate"
    echo "---"
done

echo "All hyperparameter combinations completed!"