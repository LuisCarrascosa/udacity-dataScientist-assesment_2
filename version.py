import subprocess
import sys

newLine = "\\n"


def getVersion(libs):
    """
    Description: This function prints the libraries versions.

    Arguments:
        libs: an array with the pip names of the libraries.

    Returns:
        None
    """
    versions = {
        lib: str(
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', f'{lib}'],
                    capture_output=True,
                    text=True
                )
            )
        for lib in libs
    }

    output = [
        f'{lib:15}{version.split(newLine)[1]}'
        for lib, version in versions.items()
    ]

    output.insert(0, sys.version)

    return output


versions_list = getVersion([
    'numpy',
    'pandas',
    'flask',
    'plotly',
    'scikit-learn',
    'nltk',
    'sqlalchemy',
    'joblib'
    ])

with open('versions.txt', 'w') as f:
    f.write("\n".join(versions_list))
