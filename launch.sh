env_choice=( "cartpole" )
loss_choice=( "kl" "cramer" )
num_atoms=51
num_runs=2

for env in "${env_choice[@]}"
do
    for loss in "${loss_choice[@]}" 
    do
        for run in $(seq 0 $num_runs)
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

