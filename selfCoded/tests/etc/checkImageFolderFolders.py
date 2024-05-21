import os

def check_filename(filename):
    """
    Checks if a file has "._" at the start of its filename.

    Parameters:
    filename (str): The name of the file.

    Returns:
    bool: False if the filename starts with "._", True otherwise.
    """
    return not os.path.basename(filename).startswith("._")

# Example usage:
print(check_filename("._example.txt"))  # Output: False
print(check_filename("example.txt"))    # Output: True

import os

def check_filepath(path):
    """
    Überprüft, ob eine Datei mit "._" im Dateinamen beginnt.

    Parameter:
    path (str): Der gesamte Pfad der Datei.

    Rückgabe:
    bool: False, wenn der Dateiname mit "._" beginnt, True ansonsten.
    """
    filename = os.path.basename(path)
    return not filename.startswith("._")

# Beispielaufrufe:
print(check_filepath("C:/Users/Domin/Documents/._example.txt"))  # Ausgabe: False
print(check_filepath("C:/Users/Domin/Documents/example.txt"))    # Ausgabe: True

