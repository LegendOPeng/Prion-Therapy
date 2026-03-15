import csv, os

FILES = {
    'gamma':      'gamma_edep.csv',
    'neutron':    'neutron_edep.csv',
    'carbon':     'carbon_edep.csv',
    'alpha':      'alpha_edep.csv',
}

combined = {}

for label, filepath in FILES.items():
    if not os.path.exists(filepath):
        print(f'WARNING: {filepath} not found, skipping')
        continue
    with open(filepath) as f:
        for row in csv.reader(f):
            if len(row) < 4: continue
            try:
                ix, iy, iz = int(row[0]), int(row[1]), int(row[2])
                energy = float(row[3])
                if energy > 0:
                    key = (ix, iy, iz)
                    combined[key] = combined.get(key, 0) + energy
            except ValueError:
                continue
    print(f'Loaded: {label}')

with open('combined_edep.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for (ix, iy, iz), energy in sorted(combined.items()):
        writer.writerow([ix, iy, iz, energy, 0, 1])

print(f'Done! {len(combined)} voxels -> combined_edep.csv')
