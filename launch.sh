env_choice=( "pong" )
loss_choice=( "cramer" )
num_atoms_choice=( 25 51 )
num_runs=10

for env in "${env_choice[@]}"
do
    for loss in "${loss_choice[@]}" 
    do
        for num_atoms in "${num_atoms_choice[@]}"
        do
            for run in $(seq 1 $num_runs)
            do
                path="experiments/${env}_${num_atoms}_${loss}/${run}"
                echo $path
                python main.py --loss=$loss           \
                               --env=$env             \
                               --num_atoms=$num_atoms \
                               --exp_path=$path       
            done
        done
    done
done

