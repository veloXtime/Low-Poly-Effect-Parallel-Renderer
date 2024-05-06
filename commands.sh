## test
./main ../images/emma.png

## generate tar
tar --exclude='./src/images' --exclude='./.git' --exclude='./.vscode' --exclude='./reports'  -cvzf low-poly-effect-parallel-renderer.tgz .