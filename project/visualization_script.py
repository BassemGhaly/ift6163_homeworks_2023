import pickle
import os


# Use the loaded data
import matplotlib.pyplot as plt

def plot_key(key):
    with open('//home/cyrille/Desktop/IFT6163/Project/ift6163_homeworks_2023/project/saved_dictionary_copy.pkl', 'rb') as f:
    # Load the contents of the pickle file into a variable
        data = pickle.load(f)
    # Extract the values for the given key from all dictionaries
    values = [entry[key] for entry in data.values()]

    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.plot(list(data.keys()), values, marker='o', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel(key)
    plt.title(f'{key} over Iterations')
    plt.grid(True)
    
    # Create the save directory if it doesn't exist
    save_dir = '/home/cyrille/Desktop/IFT6163/Project/ift6163_homeworks_2023/project/figures'
    if not os.path.exists('/home/cyrille/Desktop/IFT6163/Project/ift6163_homeworks_2023/project/figures'):
        os.makedirs(save_dir)

    # Save the figure to the specified directory
    save_path = os.path.join(save_dir, f'{key}_plot.png')
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

    # Display the plot in the notebook (optional)
    plt.show()

plot_key('Eval_AverageReturn')
plot_key('Training_Loss')


