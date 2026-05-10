# Interactive Viewer

The viewer is the SIBR remote-Gaussian app from [3D-GS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip).
Extract the zip into the project root:

```
├── 4d-neural-voxel-splatting/
│   ├── viewers/
│   │   ├── bin/
│   │   ├── resources/
│   │   └── shaders/
│   ├── train.py
│   ├── test.py
│   └── ...
```

## Using the viewer

If training locally:
```bash
./viewers/bin/SIBR_remoteGaussian_app.exe --port 6017  # match your training --port
```

If training on a remote server, set up a port forward (e.g. via the VS Code
port-forwarding panel) and connect the viewer to the forwarded port. Clone this
repo on your local machine and place the dataset alongside it:

```
├── 4d-neural-voxel-splatting/
│   ├── viewers/
│   ├── data/
│   │   └── dnerf/
│   ├── train.py
│   └── ...
```

Rendering speed is dominated by network bandwidth.
