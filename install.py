import os
import subprocess

def install_dependencies():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
    subprocess.run(['pip', 'install', 'setuptools'], check=True)

def install_cec2017():
    repo_url = "https://github.com/tilleyd/cec2017-py"
    repo_dir = "cec2017-py"
    if not os.path.exists(repo_dir):
        print(f"Cloning {repo_url}...")
        subprocess.run(['git', 'clone', repo_url], check=True)
    print(f"Installing {repo_dir}...")
    os.chdir(repo_dir)
    subprocess.run(['pip', 'install', '.'], check=True)
    os.chdir('..')

if __name__ == '__main__':
    install_dependencies()
    install_cec2017()
