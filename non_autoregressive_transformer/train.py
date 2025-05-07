import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np # For averaging in extensive eval

from .model import NonAutoregressiveTransformer
# data.py is now for algebraic expressions
from .data import get_batch, PAD_IDX, VOCAB_SIZE, MAX_SEQ_LEN, INT_TO_CHAR, CHAR_TO_INT

# Hyperparameters - some may need tuning for the new task
D_MODEL = 64
NHEAD = 4
NUM_ENCODER_LAYERS = 4 # Kept from previous, might need more for complex task
DIM_FEEDFORWARD = 128 
DROPOUT = 0.1
LEARNING_RATE = 0.0005 
BATCH_SIZE = 32 # Potentially reduce if MAX_SEQ_LEN is much larger and causes memory issues
NUM_TRAINING_STEPS = 10001 # Might need more steps for this harder task
LOG_INTERVAL = 200
EVAL_INTERVAL = 1000
EXTENSIVE_EVAL_SAMPLES = 200 # Number of samples for final evaluation

def calculate_accuracy_metrics(tgt_str_actual, pred_str, pad_char_to_strip=' '):
    """Calculates full sequence match and character accuracy."""
    core_tgt_str = tgt_str_actual.strip(pad_char_to_strip)
    core_pred_str = pred_str.strip(pad_char_to_strip)
    
    full_sequence_correct = 1 if core_tgt_str == core_pred_str else 0

    char_correct_count = 0
    # Compare char by char for the length of the shorter of the two core strings
    # This is one way; another is to penalize for length mismatch more explicitly.
    # For now, let's count matches up to the shorter length.
    # A better char accuracy would be Levenshtein distance based, normalized.
    # Current: common prefix / target_length
    len_target = len(core_tgt_str)
    len_pred = len(core_pred_str)

    for i in range(min(len_target, len_pred)):
        if core_tgt_str[i] == core_pred_str[i]:
            char_correct_count += 1
    
    char_accuracy = 0
    if len_target > 0:
        # Penalize if lengths are different after common prefix match
        # A simple way: char_correct_count / max(len_target, len_pred)
        # Or, if perfect prefix but different length, not 100%
        if len_target == len_pred and char_correct_count == len_target:
             char_accuracy = 1.0
        elif len_target > 0:
             char_accuracy = char_correct_count / len_target # fraction of target chars correctly predicted in prefix
        # if core_tgt_str == core_pred_str, this will be 1.0
        # if pred is shorter but correct prefix: e.g. tgt "abc", pred "ab" -> 2/3
        # if pred is longer but correct prefix: e.g. tgt "ab", pred "abc" -> 2/2 = 1.0 (but full_seq_correct is 0)
        # This definition prioritizes getting target chars right.

    elif len_target == 0 and len_pred == 0: # Both empty after strip (e.g. predicting only padding for empty target)
        char_accuracy = 1.0
        
    return full_sequence_correct, char_accuracy

def evaluate_model_extensively(model, device, num_samples, pad_char):
    print(f"\n--- Starting Extensive Evaluation ({num_samples} samples) ---")
    model.eval()
    total_full_sequence_correct = 0
    total_char_accuracy = 0
    
    # Ensure CHAR_TO_INT and INT_TO_CHAR are available if this function is called elsewhere
    # from .data import INT_TO_CHAR # Already imported at top level

    actual_samples_tested = 0
    for i in range(num_samples):
        # Get a single example, as get_batch might be slow if generate_single_example has many retries
        # However, for consistency, let's use get_batch with batch_size=1
        try:
            src_batch, tgt_batch = get_batch(batch_size=1)
        except Exception as e:
            print(f"Warning: Error getting batch for extensive eval, sample {i+1}: {e}")
            if actual_samples_tested > 0: # Avoid division by zero if no samples work
                 break
            else:
                print("No samples could be generated for extensive evaluation.")
                return

        src_item, tgt_item = src_batch.to(device), tgt_batch.to(device)
        actual_samples_tested +=1

        with torch.no_grad():
            prediction_logits = model(src_item) # (1, seq_len, vocab_size)
            predicted_indices = torch.argmax(prediction_logits[0], dim=-1) # (seq_len)

            # src_str_actual_eval = "".join([INT_TO_CHAR.get(idx.item(), '?') for idx in src_item[0].cpu()])
            tgt_str_actual_eval = "".join([INT_TO_CHAR.get(idx.item(), '?') for idx in tgt_item[0].cpu()])
            pred_str_eval = "".join([INT_TO_CHAR.get(idx.item(), '?') for idx in predicted_indices.cpu()])

            full_correct, char_acc = calculate_accuracy_metrics(tgt_str_actual_eval, pred_str_eval, pad_char)
            total_full_sequence_correct += full_correct
            total_char_accuracy += char_acc
        
        if (i + 1) % (num_samples // 10 if num_samples >=10 else 1) == 0:
            print(f"  Evaluated {i+1}/{num_samples} samples...")

    if actual_samples_tested == 0:
        print("No samples were successfully processed during extensive evaluation.")
        return

    avg_full_sequence_accuracy = total_full_sequence_correct / actual_samples_tested
    avg_char_accuracy = total_char_accuracy / actual_samples_tested

    print("--- Extensive Evaluation Results ---")
    print(f"Total samples evaluated: {actual_samples_tested}")
    print(f"Average Full Sequence Accuracy: {avg_full_sequence_accuracy:.2%}")
    print(f"Average Character Accuracy:     {avg_char_accuracy:.2%}")
    print("------------------------------------")
    model.train() # Set back to train mode

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pad_char_for_stripping = INT_TO_CHAR[PAD_IDX] # Get the actual pad character for stripping

    # Ensure these are from the NEW data.py (algebraic expressions)
    print(f"Vocab size: {VOCAB_SIZE}, Max seq len: {MAX_SEQ_LEN}, Pad IDX: {PAD_IDX}")

    model = NonAutoregressiveTransformer(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN, # Ensure this is from the new data.py
        padding_idx=PAD_IDX      # Ensure this is from the new data.py
    ).to(device)

    criterion = nn.CrossEntropyLoss() # PAD_IDX is now a learnable token
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    total_loss = 0
    start_time = time.time()

    print(f"Starting training for {NUM_TRAINING_STEPS-1} steps (algebraic expansion task)...")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    for step in range(1, NUM_TRAINING_STEPS):
        try:
            src_batch, tgt_batch = get_batch(batch_size=BATCH_SIZE)
        except Exception as e:
            print(f"CRITICAL: Error getting batch at step {step}: {e}. Skipping step.")
            # Potentially add a counter to stop if too many errors occur
            if step > 10 and step < BATCH_SIZE * 3: # Early in training, this could be fatal
                print("Multiple errors getting batch early in training. Aborting.")
                return
            continue # Skip this training step
            
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

        optimizer.zero_grad()
        logits = model(src_batch)
        loss = criterion(logits.view(-1, VOCAB_SIZE), tgt_batch.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if step % LOG_INTERVAL == 0:
            current_loss = total_loss / LOG_INTERVAL
            elapsed_time = time.time() - start_time
            print(f"Step {step}/{NUM_TRAINING_STEPS-1} | Loss: {current_loss:.4f} | Time: {elapsed_time:.2f}s")
            total_loss = 0
            start_time = time.time()

        if step % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                # Use the first item of the current batch for a quick check
                # (or fetch a fresh small batch for this inline eval too)
                test_src_single = src_batch[0:1]
                test_tgt_single_actual_indices = tgt_batch[0]

                prediction_logits_single = model(test_src_single)
                predicted_indices_single = torch.argmax(prediction_logits_single[0], dim=-1)

                src_str = "".join([INT_TO_CHAR.get(idx.item(), '?') for idx in test_src_single[0].cpu()])
                tgt_str_actual = "".join([INT_TO_CHAR.get(idx.item(), '?') for idx in test_tgt_single_actual_indices.cpu()])
                pred_str = "".join([INT_TO_CHAR.get(idx.item(), '?') for idx in predicted_indices_single.cpu()])
                
                full_correct, char_acc = calculate_accuracy_metrics(tgt_str_actual, pred_str, pad_char_for_stripping)

                print("--- EVALUATION (Sample) ---")
                print(f"Input:          '{src_str.strip(pad_char_for_stripping)}'")
                print(f"Target:         '{tgt_str_actual.strip(pad_char_for_stripping)}'")
                print(f"Prediction:     '{pred_str.strip(pad_char_for_stripping)}'")
                print(f"Full Seq Match:   { 'Yes' if full_correct else 'No'}")
                print(f"Char Accuracy:    {char_acc:.2%}")
                print("---------------------------")
            model.train()

    print("Training complete!")
    evaluate_model_extensively(model, device, EXTENSIVE_EVAL_SAMPLES, pad_char_for_stripping)

if __name__ == '__main__':
    train() 
