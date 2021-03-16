n=13
for ((i=0; i<6; ++i))
do
    nohup python main.py $n > cout.$n.$i &
    echo started process $i
done
