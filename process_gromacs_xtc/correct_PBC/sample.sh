#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time 4:00:00
#SBATCH --job-name PBC_TTT_SSS_RRR
#SBATCH --output=PBC_TTT_SSS_RRR-%J.out
#SBATCH --mem=10G 
#SBATCH --account=def-pomes


module load gromacs/2016.3


cd /scratch/jjhuang/projects/melittin_poration/melittin_mixed/files_for_analysis_whole

para=/home/jjhuang/melittin_poration/melittin_mixed/result_analysis/set_up_analysis/step_3_correct_PBC/parameters

path=/scratch/jjhuang/projects/melittin_poration/melittin_mixed

#para=/scratch/jjhuang/projects/melittin_poration/melittin_mixed
cont_list=( cont0time cont1time cont2time )

for numT in {TTT..TTT}
do
    cd transNum_${numT}
    for numS in {SSS..SSS}
    do
        cd transNum_${numT}_surfNum_${numS}
        for rpt in {RRR..RRR}
        do
            cd transNum_${numT}_surfNum_${numS}_rpa_${rpt}
            for cont_ in ${cont_list[@]}  
            do
                xtc=melittin_trans_${numT}_surf_${numS}_rpa_${rpt}_03_${cont_}_skip5.xtc
                gro=melittin_trans_${numT}_surf_${numS}_rpa_${rpt}_03_cont2time_rename.gro
                tpr=${path}/transNum_${numT}/transNum_${numT}_surfNum_${numS}/transNum_${numT}_surfNum_${numS}_rpa_${rpt}/melittin_trans_${numT}_surf_${numS}_rpa_${rpt}_03.tpr

                cp melittin_trans_${numT}_surf_${numS}_rpa_${rpt}_03_cont2time_rename.gro temp_gro.gro 
                cp topol.top temp_topol.top 

                sed -i -e "s|/global/scratch/jjhuang/melittin_poration/parameters/|/home/jjhuang/melittin_poration/melittin_mixed/result_analysis/set_up_analysis/step_3_correct_PBC/parameters/|g" temp_topol.top

                # get number of atoms in the gro file 
                temp_Num_Old=`sed '2q;d' temp_gro.gro`
                
                # increase the total number of atoms by 1
                temp_Num_New=$(($temp_Num_Old+1))                
                sed -i -e "2 s/^.*$/${temp_Num_New}/" temp_gro.gro

                # insert an atom of CL into the gro file
                temp_insert="27536CL      CL19184   3.252   1.539   5.422"
                sed -i -e "$ i ${temp_insert}" temp_gro.gro # this line is a bit slow
                      
                # update the number of CL into the topol file
                temp_CL_Old=`tail -n 1 temp_topol.top | tr -s ' ' | cut -d" " -f 2`
                temp_CL_New=$(($temp_CL_Old+1))
                sed -i -e "$ s/^.*$/CL   ${temp_CL_New}/" temp_topol.top
                    
                 
                gmx grompp -f ${para}/minim.mdp -c temp_gro.gro -p temp_topol.top -o melittin_trans_${numT}_surf_${numS}_rpa_${rpt}_03_${cont_}_runPBC.tpr  

                tpr_x=melittin_trans_${numT}_surf_${numS}_rpa_${rpt}_03_${cont_}_runPBC.tpr

                # melittin_trans_5_surf_1_rpa_1_03_cont2time.xtc has 201 frames, starting from 0, to 200
                mkdir frame_sets  ndx_sets
                /home/jjhuang/anaconda3/bin/python3.6 ${para}/python_add_psudo_points.py -ix ${xtc} -ig ${gro} -o ./

                echo 0 | gmx trjconv -s ${tpr} -f ${xtc} -o ./frame_sets/m2_frame_.gro -sep 

                if [ "${num}" == "cont0time" ];then
                    LoopCount=100
                else
                    LoopCount=200
                fi

                for num in `seq 0 1 ${LoopCount}`
                do
                    gmx editconf -f ./frame_sets/m2_frame_${num}.gro -o ./frame_sets/m2_frame_${num}.gro -resnr 1
                    Num_Old=`sed '2q;d' ./frame_sets/m2_frame_${num}.gro`
                    Num_New=$(($Num_Old+1))
                
                    # replace the number in the second line with a new one
                    sed -i -e "2 s/^.*$/${Num_New}/"  ./frame_sets/m2_frame_${num}.gro
                
                    PSUDO=`cat ndx_sets/index_PBC_${num}.txt`     
                    sed -i -e "$ i ${PSUDO}" ./frame_sets/m2_frame_${num}.gro    
                
                    echo "q" | gmx make_ndx -f ./frame_sets/m2_frame_${num}.gro  -o ./ndx_sets/index_PBC_${num}.ndx
                
                    echo "[ CL_PBC ]" >> ./ndx_sets/index_PBC_${num}.ndx 
                    echo ${Num_New} >> ./ndx_sets/index_PBC_${num}.ndx
                
                    num_format=$(printf "%04d" ${num})
                 
                    echo 0 | gmx trjconv -s ${tpr_x} -f ./frame_sets/m2_frame_${num}.gro -o ./frame_sets/m2_frame_${num}_nojump.gro -n ./ndx_sets/index_PBC_${num}.ndx -pbc nojump 
                    echo 22 0 | gmx trjconv -s ${tpr_x} -f ./frame_sets/m2_frame_${num}_nojump.gro -n ./ndx_sets/index_PBC_${num}.ndx -o ./frame_sets/m2_frame_${num}_nojump_c.gro -center 
                
                    echo 0 | gmx trjconv -s ${tpr_x} -f ./frame_sets/m2_frame_${num}_nojump_c.gro -n ./ndx_sets/index_PBC_${num}.ndx -o ./frame_sets/m2_frame_${num_format}_noPBC.gro -ur compact -pbc mol 
                
                    rm ./frame_sets/\#*    
                    rm ./frame_sets/m2_frame_${num}_nojump.gro
                    rm ./frame_sets/m2_frame_${num}_nojump_c.gro
                done

                gmx trjcat -f ./frame_sets/*_noPBC.gro -o ./frame_sets/melittin_trans_${numT}_surf_${numS}_rpa_${rpt}_03_${cont_}_skip5_noPBC.xtc

                cp ./frame_sets/melittin_trans_${numT}_surf_${numS}_rpa_${rpt}_03_${cont_}_skip5_noPBC.xtc ./
              
                rm -r ndx_sets frame_sets 
                rm temp_gro.gro temp_topol.top mdout.mdp
            done
            rm \#*
            cd ../
        done 
        cd ../
    done   
done


