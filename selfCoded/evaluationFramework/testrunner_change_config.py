import json
import sys

def update_nested_config(config, updates):
    for key, value in updates.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

def update_config(config_path, updates):
    # Lade die Konfigurationsdatei
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Nimm die Änderungen vor
    update_nested_config(config, updates)

    # Speichere die aktualisierte Konfigurationsdatei
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Konfigurationsdatei {config_path} wurde erfolgreich aktualisiert.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Verwendung: python update_config.py <config_path> <key1=value1> <key2=value2> ...")
        sys.exit(1)

    config_path = sys.argv[1]
    updates = dict(arg.split('=') for arg in sys.argv[2:])

    # Konvertiere numerische Werte in Zahlen
    for key, value in updates.items():
        if value.isdigit():
            updates[key] = int(value)
        elif value == "true" or value == "false":
            if value == "true":
                updates[key] = True
            else:
                updates[key] = False
        elif value == "null":
            updates[key] = None
        else:
            try:
                updates[key] = float(value)
            except ValueError:
                updates[key] = value

    update_config(config_path, updates)
