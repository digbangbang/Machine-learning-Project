import os

def load_data(directory):
    categories = ['ham', 'spam']
    data = []
    labels = []
    # Iterate through each enron directory
    for enron_dir in [d for d in os.listdir(directory) if d.startswith('enron')]:
        for label, category in enumerate(categories):
            category_dir = os.path.join(directory, enron_dir, category)
            for filename in os.listdir(category_dir):
                file_path = os.path.join(category_dir, filename)
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                    data.append(text)
                    labels.append(label)
    return data, labels


