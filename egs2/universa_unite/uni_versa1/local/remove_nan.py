import json
import math
import os
import argparse

def clean_json_file(input_file, output_file):
    """
    Process a file where each line is in the format '<file_id> {file_json}'
    and remove any fields with NaN or infinity values.
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_number, line in enumerate(f_in, 1):
            try:
                # Split the line into file_id and json content
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    print(f"Warning: Line {line_number} doesn't match expected format. Skipping.")
                    continue
                
                file_id, json_str = parts
                
                # Parse JSON
                try:
                    json_data = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {line_number}. Skipping.")
                    continue
                
                # Clean JSON (remove NaN and infinity values)
                cleaned_json = clean_nan_and_infinity(json_data)
                
                # Write cleaned data to output file
                f_out.write(f"{file_id} {json.dumps(cleaned_json)}\n")
                
            except Exception as e:
                print(f"Error processing line {line_number}: {e}")
    
    print(f"Processing complete. Cleaned data saved to {output_file}")

def clean_nan_and_infinity(obj):
    """
    Recursively remove NaN and infinity values from a JSON object.
    Works on nested dictionaries, lists, and other JSON-compatible data types.
    """
    if isinstance(obj, dict):
        return {key: clean_nan_and_infinity(value) for key, value in obj.items()
                if not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))}
    elif isinstance(obj, list):
        return [clean_nan_and_infinity(item) for item in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None  # You could also return 0 or another default value
    else:
        return obj

def parse_arguments():
    """
    Parse command line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description='Process JSON files and remove NaN/infinity values.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-i', '--input', 
                        required=True,
                        help='Path to the input file')
    
    parser.add_argument('-o', '--output',
                        default='cleaned_output.txt',
                        help='Path for the output file')
    
    parser.add_argument('--replace-with',
                        default='remove',
                        choices=['null', 'zero', 'remove'],
                        help='How to handle NaN/infinity values: replace with null, replace with zero, or remove the field')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        exit(1)
    
    # Process the file
    clean_json_file(args.input, args.output)
    
    print(f"Completed! Processed {args.input} and saved results to {args.output}")
