import os
import subprocess
import time
import sys
import glob

N = 4

def main():
    try:
        processes = set()

        notebooks = glob.glob(os.path.join("test", "**", "*.ipynb"), recursive=True)
        for file in notebooks:
            print(f"Starting {file}...")
            processes.add(subprocess.Popen(["jupyter", "nbconvert", "--execute", "--to", "html", file]))
            if len(processes) >= N:
                time.sleep(1.0)
                processes.difference_update([
                    p for p in processes if p.poll() is not None
                ])

        scripts = glob.glob(os.path.join("test", "**", "*.py"), recursive=True)
        for file in scripts:
            print(f"Starting {file}...")
            log = os.path.splitext(file)[0] + ".txt"
            processes.add(subprocess.Popen(["py", file, ">", log]))
            if len(processes) >= N:
                time.sleep(1.0)
                processes.difference_update([
                    p for p in processes if p.poll() is not None
                ])

        while any([p.poll() is None for p in processes]):
            # Still waiting
            time.sleep(1.0)

    except BaseException as e:
        print(f"Killing {len(processes)} processes due to {str(e)}")
        for p in processes:
            p.kill()

if __name__ == "__main__":
    main()