import json
import sys

def update_config(config_path, updates):
    # Lade die Konfigurationsdatei
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Nimm die Änderungen vor
    for item in config:
        if item['class'] in updates:
            for key, value in updates[item['class']].items():
                if key in item['init_params']:
                    item['init_params'][key] = value
                else:
                    print(f"Warnung: Schlüssel {key} nicht in init_params für {item['class']} gefunden.")

    # Speichere die aktualisierte Konfigurationsdatei
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Konfigurationsdatei {config_path} wurde erfolgreich aktualisiert.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Verwendung: python update_attack_config.py <config_path> <class:key=value> ...")
        sys.exit(1)

    config_path = sys.argv[1]
    updates = {}
    for arg in sys.argv[2:]:
        class_name, key_value = arg.split(':')
        key, value = key_value.split('=')
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                value = value

        if class_name not in updates:
            updates[class_name] = {}
        updates[class_name][key] = value

    update_config(config_path, updates)
