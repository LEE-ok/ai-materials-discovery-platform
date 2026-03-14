import argparse
import multiprocessing
import os
import sys
import time

# Force legacy Keras to avoid conflicts between Keras 3 and TFP
os.environ['TF_USE_LEGACY_KERAS'] = '1'

def install_dependencies():
    """Check and install missing requirements."""
    import subprocess
    import pkg_resources
    
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(requirements_path):
        return

    print("Checking dependencies...")
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    missing = []
    for requirement in requirements:
        try:
            pkg_resources.require(requirement)
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing.append(requirement)

    if missing:
        print(f"Missing libraries found: {', '.join(missing)}")
        print("Installing missing libraries (this may take a while)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Successfully installed all dependencies.")
        except Exception as e:
            print(f"Failed to install dependencies automatically: {e}")
            print("Please run 'pip install -r requirements.txt' manually.")

# Perform dependency check before any other imports that might fail
if __name__ == '__main__':
    install_dependencies()

def start_api_server():
    """Start the Flask API server."""
    print("Initializing Flask API server...")
    from src.api.app import app
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

def start_gui_client():
    """Start the PyQt6 GUI Application."""
    time.sleep(1) # Give the server a moment to start
    print("Launching PyQt6 Client GUI...")
    from src.gui.main_window import launch_gui
    launch_gui()

def main():
    parser = argparse.ArgumentParser(description="AI Materials Discovery Platform")
    parser.add_argument('--mode', type=str, choices=['api', 'gui', 'all'], default='all',
                        help='Mode to run the platform in: api, gui, or both (all)')
    args = parser.parse_args()

    # Prepend the project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if args.mode == 'api':
        start_api_server()
    elif args.mode == 'gui':
        start_gui_client()
    elif args.mode == 'all':
        # Start both using multiprocessing
        api_process = multiprocessing.Process(target=start_api_server)
        api_process.start()

        try:
            start_gui_client()
        finally:
            print("Shutting down API server...")
            api_process.terminate()
            api_process.join()

if __name__ == '__main__':
    # Set the multiprocessing start method for cross-platform compatibility
    if sys.platform == "win32":
        multiprocessing.freeze_support()
        
    try:
        main()
    except Exception as e:
        import traceback
        with open("final_error.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        print(f"Fatal error occurred: {e}")
        traceback.print_exc()
