from backend.metrics import compute_all_metrics

# Using the exact same file that we know loads perfectly
test_file = "test_data/audios/bengali/train-00000-of-00096/08c84a1d-646f-419e-926b-ac95c35ca19d_1_chunk_11.flac"

print("Running full metrics pipeline on one file...")

# We will call the function exactly as your processor does
results = compute_all_metrics(test_file)

print("\n" + "="*50)
print("PIPELINE RESULTS:")
for key, value in results.items():
    print(f"{key}: {value}")
print("="*50 + "\n")