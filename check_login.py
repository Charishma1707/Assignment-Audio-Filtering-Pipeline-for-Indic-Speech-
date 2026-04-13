from huggingface_hub import whoami

print("Checking Hugging Face login status...")

try:
    user_info = whoami()
    print("✅ SUCCESS! You are logged in as:", user_info['name'])
except Exception as e:
    print("❌ NOT LOGGED IN. Here is the error:")
    print(e)