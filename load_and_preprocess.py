import pandas as pd

def load_fasttext_file(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split label and text
            label, text = line.split(' ', 1)
            # convert __label__1/__label__2 to 0/1
            label = int(label.replace("__label__", "")) - 1
            texts.append(text)
            labels.append(label)
    return pd.DataFrame({"reviewText": texts, "sentiment": labels})

if __name__ == "__main__":
    train_df = load_fasttext_file(r"C:\Users\Prathibha KS\Desktop\AMSS Project\train.ft.txt\train.ft.txt")
    test_df = load_fasttext_file(r"C:\Users\Prathibha KS\Desktop\AMSS Project\test.ft.txt\test.ft.txt")

    # Save as CSV for easy use in modeling file
    train_df.to_csv("train_preprocessed.csv", index=False)
    test_df.to_csv("test_preprocessed.csv", index=False)
    
    print("Preprocessing done. Train and Test CSVs created.")
