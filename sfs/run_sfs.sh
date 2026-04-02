#!/bin/bash

# Ensure exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {cube|real}"
    exit 1
fi

MODE="$1"
PYTHON_SCRIPT="sfs/sfs.py"

case "$MODE" in
    real)
        echo "Running SfS on real data..."
        python3 "$PYTHON_SCRIPT" media/testing/test_real_data/objs media/testing/test_real_data/masks media/testing/test_real_data/real_recon.ply media/testing/test_real_data/projs.npz -s
        ;;
    
    cube)
        echo "Running SfS on cube data..."
        python3 "$PYTHON_SCRIPT" media/testing/test_cube/imgs media/testing/test_cube/masks media/testing/test_cube/cube_recon.ply media/testing/test_cube/projs.npz
        ;;

	torus)
		echo "Running SfS on torus data..."
		python3 "$PYTHON_SCRIPT" media/testing/test_torus/imgs media/testing/test_torus/masks media/testing/test_torus/torus_recon.ply media/testing/test_torus/projs.npz
		;;

	ellipsoid)
		echo "Running SfS on ellipsoid data..."
		python3 "$PYTHON_SCRIPT" media/testing/test_ellipsoid/imgs media/testing/test_ellipsoid/masks media/testing/test_ellipsoid/ellipsoid_recon.ply media/testing/test_ellipsoid/projs.npz
		;;
    
    *)
        echo "Invalid option: $MODE"
        echo "Usage: $0 {cube|real}"
        exit 1
        ;;
esac
