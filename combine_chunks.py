import os
import glob

def concat_chunks(chunk_pattern, output_file):
    # Get a sorted list of chunk files
    chunks = sorted(glob.glob(chunk_pattern))
    if not chunks:
        print("No chunk files found matching:", chunk_pattern)
        return

    # Open the final output file in append binary mode.
    with open(output_file, "ab") as outfile:
        for chunk in chunks:
            print(f"Processing {chunk}...")
            try:
                with open(chunk, "rb") as infile:
                    # Read and write in 1MB blocks
                    while True:
                        data = infile.read(1024 * 1024)
                        if not data:
                            break
                        outfile.write(data)
                # Flush to ensure data is written
                outfile.flush()
            except Exception as e:
                print(f"Error processing {chunk}: {e}")
                return

            try:
                os.remove(chunk)
                print(f"Deleted {chunk} after appending.")
            except Exception as e:
                print(f"Error deleting {chunk}: {e}")
                return

    print(f"All chunks concatenated into {output_file}.")

if __name__ == "__main__":
    chunk_pattern = "yfcc100m_phone.zip.rclone_chunk.*"
    output_file = "yfcc100m_phone.zip"
    concat_chunks(chunk_pattern, output_file)
