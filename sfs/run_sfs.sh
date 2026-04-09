#!/bin/bash

# Ensure exactly two arguments are provided
if [ "$#" -ne 2 ]; then
	echo "Usage: $0 {bunny|cube|torus|ellipsoid} {halfspace|volumetric}"
	exit 1
fi

MODE="$1"
METHOD="$2"
PYTHON_SCRIPT="sfs/sfs.py"

# Map method to Python flag
if [ "$METHOD" == "halfspace" ]; then
	METHOD_FLAG="halfspace"
elif [ "$METHOD" == "volumetric" ]; then
	METHOD_FLAG="volumetric"
else
	echo "Invalid method: $METHOD"
	exit 1
fi

echo "Running $METHOD SfS on $MODE data..."

case "$MODE" in
	bunny)
		python3 "$PYTHON_SCRIPT" \
			media/bunny_data/images \
			media/bunny_data/silhouettes \
			media/bunny_data/bunny_recon_$METHOD \
			media/bunny_data/projs.npz \
			$METHOD_FLAG
		;;

	cube)
		python3 "$PYTHON_SCRIPT" \
			media/testing/test_cube/imgs \
			media/testing/test_cube/masks \
			media/testing/test_cube/cube_recon_$METHOD \
			media/testing/test_cube/projs.npz \
			$METHOD_FLAG
		;;

	torus)
		python3 "$PYTHON_SCRIPT" \
			media/testing/test_torus/imgs \
			media/testing/test_torus/masks \
			media/testing/test_torus/torus_recon_$METHOD \
			media/testing/test_torus/projs.npz \
			$METHOD_FLAG
		;;

	ellipsoid)
		python3 "$PYTHON_SCRIPT" \
			media/testing/test_ellipsoid/imgs \
			media/testing/test_ellipsoid/masks \
			media/testing/test_ellipsoid/ellipsoid_recon_$METHOD \
			media/testing/test_ellipsoid/projs.npz \
			$METHOD_FLAG
		;;

	*)
		echo "Invalid option: $MODE"
		exit 1
		;;
esac
