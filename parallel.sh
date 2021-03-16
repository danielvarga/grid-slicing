n=12
for ((i=0; i<6; ++i))
do
    python main.py $n > cout.$i &
    echo started process $i
done
