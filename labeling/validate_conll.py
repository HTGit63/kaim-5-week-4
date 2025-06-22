# labeling/validate_conll.py

import sys

def validate(path: str):
    with open(path, encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f]
    errors = []
    for i, line in enumerate(lines, 1):
        if not line:
            continue  # blank lines separate sentences
        parts = line.split()
        if len(parts) != 2:
            errors.append(f"Line {i}: should have 2 columns, got {len(parts)}")
        token, tag = parts
        if tag != "O" and not tag.startswith(("B-", "I-")):
            errors.append(f"Line {i}: invalid tag '{tag}'")
    if errors:
        print("❌ Validation errors found:")
        for e in errors:
            print(" ", e)
    else:
        print("✅ No formatting errors in CoNLL file.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_conll.py path/to/labeled_data.conll.txt")
    else:
        validate(sys.argv[1])
