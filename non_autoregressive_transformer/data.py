import torch
import random
import sympy

# Define vocabulary for algebraic expressions
# PAD_CHAR must be first for PAD_IDX = 0
PAD_CHAR = ' '
_possible_vars = "abcde"
_digits = "0123456789"
_operators_and_parens = "*+()"
# Consider if sympy might output '**' for power. If so, that needs careful tokenization or a replacement strategy.
# For now, assuming it won't be common for simple a*(b+c) type or that str() won't use it much over repeated multiplication.
VOCAB = [PAD_CHAR] + list(_possible_vars) + list(_digits) + list(_operators_and_parens)
CHAR_TO_INT = {char: i for i, char in enumerate(VOCAB)}
INT_TO_CHAR = {i: char for i, char in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = CHAR_TO_INT[PAD_CHAR]

MAX_SEQ_LEN = 12 

# Define sympy symbols
_SYMPY_VARS = [sympy.symbols(var) for var in _possible_vars]
_VAR_MAP = {s.name: s for s in _SYMPY_VARS}

def _sympy_to_string(expr):
    """Converts a sympy expression to a string with explicit '*' and no spaces."""
    # str(expr) can sometimes produce "a*b + a*c"
    # sstr(expr, order='lex') gives a canonical representation but might be too strict
    # For now, simple str() and then replace spaces and ensure explicit multiplication if needed.
    # sympy.printing.str.StrPrinter settings can also be used.
    s = str(expr).replace(' ', '') # Remove spaces
    # Ensure explicit multiplication between variables or var and parenthesis
    # e.g. ab -> a*b, a( -> a*(. This is harder to do robustly with string replacement.
    # Using sympy's printing capabilities or ensuring expressions are built with explicit ops is better.
    # For now, our generator will build expressions carefully.
    # A simple check: if 'ab' in s, it's an issue.
    # The way we construct expressions below should yield explicit '*' from sympy.
    return s

def generate_random_expression(max_vars=3, max_ops=3):
    """
    Generates a random simple algebraic expression using sympy.
    Ensures expressions are like a*(b+c) or (a+b)*c etc.
    Returns a tuple (sympy_expr, was_multi_term_factor).
    was_multi_term_factor helps identify expressions that are suitable for expansion.
    """
    num_vars_in_expr = random.randint(2, max_vars)
    vars_to_use = random.sample(_SYMPY_VARS, num_vars_in_expr)
    
    expr = None
    was_multi_term_factor = False # Indicates if a factor like (b+c) was created

    # Try to create expressions that are interesting to expand
    # Pattern 1: var1 * (var2 + var3)
    if num_vars_in_expr >= 2 and random.random() < 0.7: # Higher chance for this structure
        op_type = random.choice([sympy.Add, sympy.Mul])
        
        # Ensure at least one factor is a sum for expansion
        use_sum_factor = random.choice([True, False]) if op_type == sympy.Mul else True

        if op_type == sympy.Mul and num_vars_in_expr >= 2:
            term1 = vars_to_use[0]
            if use_sum_factor and num_vars_in_expr >= 3:
                term2 = sympy.Add(vars_to_use[1], vars_to_use[2], evaluate=False)
                was_multi_term_factor = True
            elif use_sum_factor and num_vars_in_expr == 2: # e.g. a*(a+b)
                 term2 = sympy.Add(vars_to_use[0], vars_to_use[1], evaluate=False)
                 was_multi_term_factor = True
            else: # e.g. a*b or a*a
                term2 = vars_to_use[1]
            expr = sympy.Mul(term1, term2, evaluate=False)
        elif op_type == sympy.Add and num_vars_in_expr >= 2 : # Simple sum a+b, not much to expand unless part of larger
            expr = sympy.Add(vars_to_use[0], vars_to_use[1], evaluate=False)
        elif num_vars_in_expr >= 1: # Fallback to single var if other structures fail
            expr = vars_to_use[0]
        else: # Should not happen if num_vars_in_expr >= 2
             return vars_to_use[0], False


    # Pattern 2: (var1 + var2) * var3  -- this is covered by var1*(var2+var3) due to commutativity of Mul
    # Let's try to make more (a+b)*(c+d) type or (a+b)*c
    elif num_vars_in_expr >= 3 and random.random() < 0.5 : # (a+b)*c or (a+b)*(c+d)
        if num_vars_in_expr >= 4 and random.random() < 0.5: # (a+b)*(c+d)
            term1 = sympy.Add(vars_to_use[0], vars_to_use[1], evaluate=False)
            term2 = sympy.Add(vars_to_use[2], vars_to_use[3], evaluate=False)
            expr = sympy.Mul(term1, term2, evaluate=False)
            was_multi_term_factor = True
        else: # (a+b)*c
            term1 = sympy.Add(vars_to_use[0], vars_to_use[1], evaluate=False)
            term2 = vars_to_use[2]
            expr = sympy.Mul(term1, term2, evaluate=False)
            was_multi_term_factor = True
    
    if expr is None: # Fallback if no complex structure was made
        if num_vars_in_expr >= 2:
            expr = sympy.Add(vars_to_use[0], vars_to_use[1], evaluate=False)
        else:
            expr = vars_to_use[0]
            
    return expr, was_multi_term_factor


def generate_single_example(min_expansion_diff=2):
    """
    Generates a single data example for the algebraic expansion task.
    Input: unexpanded expression string
    Target: expanded expression string
    Returns:
        (torch.Tensor, torch.Tensor): Input tensor and target tensor.
        Or (None, None) if a suitable example couldn't be generated.
    """
    for _attempt in range(20): # Try a few times to get a good example
        unexpanded_expr_sympy, suitable_for_expansion = generate_random_expression()
        
        input_str_core = _sympy_to_string(unexpanded_expr_sympy)
        
        # Expand the expression
        expanded_expr_sympy = sympy.expand(unexpanded_expr_sympy)
        target_str_core = _sympy_to_string(expanded_expr_sympy)

        # Filter: ensure expansion actually changed the string significantly
        # and that it was of a type we expect to expand
        if not suitable_for_expansion and input_str_core == target_str_core :
            continue # Try again if not suitable and no change
        if len(target_str_core) < len(input_str_core) + min_expansion_diff and input_str_core != target_str_core and suitable_for_expansion:
            pass # Allow if it expanded, even if shorter, if it was a suitable type
        elif len(target_str_core) < len(input_str_core) + min_expansion_diff :
            continue


        # Filter out expressions that are too long or too short
        if not (2 < len(input_str_core) <= MAX_SEQ_LEN and 2 < len(target_str_core) <= MAX_SEQ_LEN):
            continue

        # Final check for unwanted characters (e.g. if sympy introduces something outside vocab)
        # This should ideally be handled by better sympy_to_string or expression construction
        if any(c not in CHAR_TO_INT for c in input_str_core) or \
           any(c not in CHAR_TO_INT for c in target_str_core):
            print(f"Warning: Generated string with chars outside vocab. Input: '{input_str_core}', Target: '{target_str_core}'. Skipping.")
            continue
            
        # Pad sequences
        padded_input_str = input_str_core.ljust(MAX_SEQ_LEN, PAD_CHAR)
        padded_target_str = target_str_core.ljust(MAX_SEQ_LEN, PAD_CHAR)

        # Convert to integer token sequences
        input_tokens = [CHAR_TO_INT[char] for char in padded_input_str]
        target_tokens = [CHAR_TO_INT[char] for char in padded_target_str]
        
        return torch.tensor(input_tokens, dtype=torch.long), \
               torch.tensor(target_tokens, dtype=torch.long)
    
    # print("Failed to generate a suitable example after multiple attempts.")
    return None, None # Could not generate a suitable example

def get_batch(batch_size: int):
    input_batch = []
    target_batch = []
    generated_count = 0
    while generated_count < batch_size:
        inp_example, tar_example = generate_single_example()
        if inp_example is not None and tar_example is not None:
            input_batch.append(inp_example)
            target_batch.append(tar_example)
            generated_count += 1
    
    return torch.stack(input_batch), torch.stack(target_batch)

if __name__ == '__main__':
    print(f"Vocabulary (size {VOCAB_SIZE}): {VOCAB}")
    print(f"Char to Int mapping: {CHAR_TO_INT}")
    print(f"Padding index: {PAD_IDX}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")

    print("\n--- Generating Test Examples (Algebraic Expansion) ---")
    success_count = 0
    for i in range(10): # Print 10 examples
        inp_ex, tar_ex = generate_single_example()
        if inp_ex is not None:
            success_count +=1
            inp_ex_str = "".join([INT_TO_CHAR[idx.item()] for idx in inp_ex])
            tar_ex_str = "".join([INT_TO_CHAR[idx.item()] for idx in tar_ex])
            print(f"Example {success_count}:")
            print(f"  Input:  '{inp_ex_str.strip()}' (len {len(inp_ex_str.strip())})")
            print(f"  Target: '{tar_ex_str.strip()}' (len {len(tar_ex_str.strip())})")
            # Verify with sympy again
            try:
                s_inp = sympy.parse_expr(inp_ex_str.strip(), local_dict=_VAR_MAP, transformations='all')
                s_expanded = sympy.expand(s_inp)
                s_expanded_str = _sympy_to_string(s_expanded)
                print(f"  Sympy Check (Target vs Sympy(Input)): '{tar_ex_str.strip()}' vs '{s_expanded_str}' -> {'Match' if tar_ex_str.strip() == s_expanded_str else 'MISMATCH'}")

            except Exception as e:
                print(f"  Error during sympy re-check: {e}")

        if i == 9 and success_count == 0:
            print("Failed to generate any valid examples in the test run.")
    
    if success_count > 0:
        print("\n--- Batch of Data (batch_size=3) ---")
        batch_inputs, batch_targets = get_batch(batch_size=3)
        print(f"Batch Inputs shape: {batch_inputs.shape}")
        print(f"Batch Targets shape: {batch_targets.shape}")
        print("Example from batch:")
        inp_ex_str = "".join([INT_TO_CHAR[idx.item()] for idx in batch_inputs[0]])
        tar_ex_str = "".join([INT_TO_CHAR[idx.item()] for idx in batch_targets[0]])
        print(f"  Input:  '{inp_ex_str.strip()}'")
        print(f"  Target: '{tar_ex_str.strip()}'")
    else:
        print("\nCould not generate a batch because no single examples were successful.") 