import h5py

with h5py.File('/home/kgupta/data/registration_testing/h5_files/RED_DSB_trainsplit.h5', 'r') as f:
    for phase in ['Train','Test','Val']:
        pg = f[f'Patches/{phase}']
        print(phase, 'number of patches:')
        print(len(list(pg.keys())))

    print('Metadata keys')
    print(list(f['Metadata'].keys()))
