# schaeffer + havard oxford parcellation
atlas_pkl = pd.read_pickle(f'{base_dir}/masks/atlases/Schaefer100_HO-subcort25_1mm.pkl')
atlas_img = atlas_pkl['image']
nilearn.plotting.plot_roi(atlas_img, title='Combined Schaefer & HO atlas', draw_cross=False, colorbar=True, cmap='Paired')
plt.show()