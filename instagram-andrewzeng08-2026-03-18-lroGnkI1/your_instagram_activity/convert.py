import json
import os

# ==========================================
# 1. CONFIGURATION (Update these variables!)
# ==========================================

# Replace 'friend_folder_name' with the exact folder name of the chat you want to use
CHAT_FOLDER_NAME = "sarashrivastava_17879363133350739" 

# The path based on your screenshot
INPUT_FILE = os.path.join(
    "messages", 
    "inbox", 
    CHAT_FOLDER_NAME, 
    "message_1.json"
)  

OUTPUT_FILE = "vertex_tuning_data.jsonl"

# Open message_1.json and copy the exact "sender_name" strings.
PROMPT_SENDER = "Andrew Zeng"      # The person asking/prompting (Vertex "user")
TARGET_SENDER = "Sara Shrivastava"  # The person responding (Vertex "model")

# ==========================================
# 2. PROCESSING LOGIC
# ==========================================

def process_instagram_chat():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}. Check your folder name!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = data.get("messages", [])

    
    # 1. Sort chronologically (Meta exports newest-first, AI needs oldest-first)
    messages.sort(key=lambda x: x.get("timestamp_ms", 0))
    
    grouped_messages = []
    
    # 2. Extract text, fix encoding, and group consecutive messages
    for msg in messages:
        # Skip photos, reels, missed calls, and reactions (text only)
        if "content" not in msg:
            continue
            
        sender = msg.get("sender_name")
        raw_content = msg.get("content")
        
        # Fix Meta's broken text encoding (Mojibake)
        try:
            content = raw_content.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            content = raw_content
            
        # Combine rapid-fire messages from the same sender into one block
        if grouped_messages and grouped_messages[-1]["sender"] == sender:
            grouped_messages[-1]["content"] += "\n" + content
        else:
            grouped_messages.append({"sender": sender, "content": content})

    # 3. Create the Input/Output pairs for Vertex AI
    tuning_examples = []
    for i in range(len(grouped_messages) - 1):
        if grouped_messages[i]["sender"] == PROMPT_SENDER and grouped_messages[i+1]["sender"] == TARGET_SENDER:
            example = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": grouped_messages[i]["content"]}]
                    },
                    {
                        "role": "model",
                        "parts": [{"text": grouped_messages[i+1]["content"]}]
                    }
                ]
            }
            tuning_examples.append(example)

    # 4. Write to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for example in tuning_examples:
            f.write(json.dumps(example) + '\n')
            
    print(f"Success! Generated {len(tuning_examples)} training examples in {OUTPUT_FILE}.")

if __name__ == "__main__":
    process_instagram_chat()