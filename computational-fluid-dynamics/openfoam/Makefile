openfoam-install:
	sh bin/install.sh

openfoam-run:
	openfoam-docker -dry-run
	openfoam-docker

openfoam-generate-mesh:
	blockMesh
openfoam-solve-incompressible-laminar-navier-stokes-equations-using-piso-algorithm:
	icoFoam
openfoam-generate-openfoam-file:
	paraFoam -touch
