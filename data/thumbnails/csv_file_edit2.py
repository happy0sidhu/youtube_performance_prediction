import pandas as pd

def remove_extensions_from_csv(input_csv, output_csv=None):
    """
    Removes file extensions from all elements in the first column of a CSV file.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str, optional): Path to save modified CSV. If None, overwrites input file.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Remove extensions from first column
    first_col = df.columns[0]
    df[first_col] = df[first_col].str.replace(r'\.\w+$', '', regex=True)
    
    # Save the modified DataFrame
    if output_csv is None:
        output_csv = input_csv
    df.to_csv(output_csv, index=False)
    print(f"File extensions removed. Saved to: {output_csv}")

# Example usage
if __name__ == "__main__":
    input_file = "face_detection_results.csv"  # Replace with your CSV path
    output_file = "modified_file.csv"  # Set to None to overwrite original
    
    remove_extensions_from_csv(input_file, output_file)