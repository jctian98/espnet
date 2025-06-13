import os
import argparse
from kaldiio import ReadHelper
import soundfile as sf

def convert_to_flac_and_create_scp(input_scp, output_dir, output_scp):
    """
    Convert audio files specified in a Kaldi wav.scp file to individual FLAC files
    and create a new wav.scp file with updated absolute paths.
    
    Args:
        input_scp: Path to the original wav.scp file
        output_dir: Directory to save FLAC files
        output_scp: Path to the new wav.scp file to create
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the output scp file for writing
    with open(output_scp, 'w') as scp_out:
        # Process each entry in the wav.scp file
        with ReadHelper(f'scp:{input_scp}') as reader:
            total_files = 0
            for key, (rate, numpy_array) in reader:
                # Define output file path
                output_file = os.path.join(output_dir, f"{key}.flac")
                
                # Get absolute path for the new scp file
                abs_path = os.path.abspath(output_file)

                if os.path.exists(abs_path):
                    total_files += 1
                    continue
                
                # Save as FLAC file using soundfile
                sf.write(output_file, numpy_array, rate, format='FLAC')
                
                # Write to the new wav.scp file
                scp_out.write(f"{key} {abs_path}\n")
                
                total_files += 1
                if total_files % 100 == 0:
                    print(f"Processed {total_files} files...", flush=True)
    
    print(f"Successfully converted {total_files} files to FLAC format in {output_dir}")
    print(f"Created new wav.scp file at {output_scp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Kaldi audio files to FLAC format and create new wav.scp')
    parser.add_argument('input_scp', help='Path to the original wav.scp file')
    parser.add_argument('output_dir', help='Directory to save FLAC files')
    parser.add_argument('output_scp', help='Path to the new wav.scp file to create')
    
    args = parser.parse_args()
    convert_to_flac_and_create_scp(args.input_scp, args.output_dir, args.output_scp)
