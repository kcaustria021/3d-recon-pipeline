# Multi-view Shape-from-Silhouette Reconstruction

This repository contains code used in completion of a class project that investigates
two different SfS paradigms.

To run the reconstruction, choose an object from ["cube", "ellipsoid", "torus", "bunny"], then the SfS mode from ["volumetric", "halfspace"].

Then, simply run:
```
sfs/run_sfs.sh {object} {mode}
```

To visualize the reconstruction, simply run:
```
python3.xx viz.py {path}
```
where `path` is the path to the ply file you want to view.