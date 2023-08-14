import os
from transformers import BartTokenizer, BartForConditionalGeneration

# Root directory containing the text files
root_dir = 'topkreate_kreatewebsite'

# Find txt files in a root_dir and read them
def find_and_read_txt_files(root_dir):
    txt_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    content = txt_file.read()
                    txt_files.append((txt_path, content))
    
    return txt_files

txt_files = find_and_read_txt_files(root_dir)

# Initialize tokenizer and model from pretrained weights
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Generate summary for each txt file
for txt_path, txt_content in txt_files:
    # Tokenize the text
    inputs = tokenizer.batch_encode_plus([txt_content], max_length=1024, return_tensors='pt', truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, min_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Save summary in the same directory as the original txt file
    summary_filename = os.path.splitext(os.path.basename(txt_path))[0] + '_summary.comments' 
    summary_path = os.path.join(os.path.dirname(txt_path), summary_filename)
    
    # Overwrite existing summary file if it exists
    if os.path.exists(summary_path):
        os.remove(summary_path)
    
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(summary)

    print(f"Summary saved for {txt_path} at {summary_path}")
\