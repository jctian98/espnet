import json
import re
import argparse

def process_file(input_filename, output_filename):
    # Open the input file for reading and the output file for writing
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        # Process each line in the file
        for line_number, line in enumerate(infile, 1):
            try:
                # Extract the identifier and JSON parts
                # The pattern looks for any text followed by a JSON object
                match = re.match(r'^(.*?)(\{.*\})$', line.strip())
                
                if match:
                    prefix = match.group(1).strip()
                    json_str = match.group(2)
                    
                    # Parse the JSON string
                    data = json.loads(json_str)
                    
                    # Remove the "key" field if it exists
                    if "key" in data:
                        del data["key"]
                    
                    if "whisper_hyp_text" in data:
                        del data["whisper_hyp_text"]
                    if "match_details" in data:
                        del data["match_details"]
                     
                    # Convert back to JSON string
                    new_json_str = json.dumps(data)
                    
                    # Write the modified line to the output file
                    outfile.write(f"{prefix} {new_json_str}\n")
                else:
                    # If the line doesn't match the expected format, copy it as is
                    print(f"Warning: Line {line_number} does not match expected format. Copying as is.")
                    outfile.write(line)
                    
            except json.JSONDecodeError:
                # If there's an error parsing the JSON, copy the line as is
                print(f"Error: Could not parse JSON in line {line_number}. Copying as is.")
                outfile.write(line)
    
    print(f"Processing complete. Output written to {output_filename}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process a file to remove "key" field from JSON objects.')
    parser.add_argument('input_file', help='Path to the input file to process')
    parser.add_argument('output_file', help='Path to the output file to write results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output about processing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Display info if verbose mode is on
    if args.verbose:
        print(f"Processing file: {args.input_file}")
        print(f"Output will be written to: {args.output_file}")
    
    # Process the file
    process_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
