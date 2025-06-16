import json
import argparse

def convert_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    # Parse the JSON object
                    data = json.loads(line.strip())
                    
                    # Extract the key
                    key = data.get("key", "")
                    
                    if key:
                        # Create the new format: key + original JSON
                        new_line = f"{key} {line.strip()}\n"
                        f_out.write(new_line)
                    else:
                        # Handle case where "key" is missing
                        print(f"Warning: 'key' not found in line: {line[:50]}...")
                        f_out.write(line)  # Write the original line
                except json.JSONDecodeError:
                    print(f"Error parsing JSON in line: {line[:50]}...")
                    # Write the original line in case of error
                    f_out.write(line)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Convert JSONL files to have keys at the front of each line.'
    )
    
    # Add arguments
    parser.add_argument('input_file', 
                        help='Path to the input JSONL file')
    parser.add_argument('output_file', 
                        help='Path to the output file')
    parser.add_argument('--key-field', default='key',
                        help='The JSON field to use as the key (default: "key")')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress warning messages')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the file
    with open(args.input_file, 'r', encoding='utf-8') as f_in:
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            line_count = 0
            error_count = 0
            
            for line in f_in:
                line_count += 1
                if not line.strip():
                    continue
                
                try:
                    # Parse the JSON object
                    data = json.loads(line.strip())
                    
                    # Extract the key using the specified field
                    key = data.get(args.key_field, "")
                    
                    if key:
                        # Create the new format: key + original JSON
                        new_line = f"{key} {line.strip()}\n"
                        f_out.write(new_line)
                    else:
                        error_count += 1
                        if not args.quiet:
                            print(f"Warning: '{args.key_field}' not found in line {line_count}: {line[:50]}...")
                        f_out.write(line)  # Write the original line
                except json.JSONDecodeError:
                    error_count += 1
                    if not args.quiet:
                        print(f"Error parsing JSON in line {line_count}: {line[:50]}...")
                    # Write the original line in case of error
                    f_out.write(line)
            
            # Print summary
            print(f"Processed {line_count} lines with {error_count} errors")
            print(f"Conversion complete: {args.input_file} → {args.output_file}")

if __name__ == "__main__":
    main()
