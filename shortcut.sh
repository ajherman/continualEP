scancel -u ajherman
rm -rf *old*
salloc -N 12 -p shared-gpu
./main.sh
