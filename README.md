# TreeDRSparsing
The codes for the paper "Discourse Representation Parsing for Sentences and Documents" ACL 2019.

## The requirements

    python 2.7
    pytorch > 0.4.1
  
## Prepare
Clone the codes on the branch DRTS

    git clone -b bs_sattn_drssup https://github.com/LeonCrashCode/TreeDRSparsing.git
    cd TreeDRSparsing/workspace/gd_sys1
    
Download the dataset from https://drive.google.com/open?id=1tuKOXIKdplDUM8EfFSZUc-WjeV7da9ZH, move them to the folder data/; Download the pretrained vector from https://drive.google.com/open?id=1ICyISR-0PhuQYxIsqE5P7_r-OCsETIEU, move it to the folder embeddings/ (ensure each line is a word representation).

    cd data
    data/trn.tree.align_drs
    data/dev.tree.align_drs
    data/tst.tree.align_drs
    data/dict
    
    ln -s ../../../scripts/tree2oracle_doc_drs.py
    
    python tree2oracle_doc_drs.py trn.tree.align_drs
    python tree2oracle_doc_drs.py dev.tree.align_drs
    python tree2oracle_doc_drs.py tst.tree.align_drs
    
The oracles are got as:
  
    data/trn.tree.align_drs.oracle.doc.*
    data/dev.tree.align_drs.oracle.doc.*
    data/tst.tree.align_drs.oracle.doc.*
    
## Training
Using the command below for training:

    cd ..  # to the folder TreeDRSparsing/workspace/gd_sys1
    ln -s ../../DRS_config
    bash train.sh
    
The models are save in the foler models, for each checkpoint, a model will be saved e.g. model1.

## Evaluation
For each checkpoint, we need to see the F1 score on development dataset

    mkdir dev_outputs
    bash bash_dev.sh
    
    cd dev_outputs
    ln -s ../data/dev.tree.align_drs.oracle.doc.in
    ln -s ../data/dev.tree.align_drs.oracle.doc.out
    ln -s ../../scripts/oracle2tree_drs.py
    ln -s ../../scripts/oracle2tree_drs.sh
    ln -s ../../scripts/drs2tuple.py
    ln -s ../../scripts/drs2tuple.sh
    ln -s ../../scripts/D-match
 
    python oracle2tree_drs.py dev.tree.align_drs.oracle.doc.in dev.tree.align_drs.oracle.doc.out > dev.tree.align_drs.gold
    python drs2tuple.py dev.tree.align_drs.gold > dev.tuple.align_drs.gold
    bash oracle2tree_drs.sh [start_checkpoint_index] [end_checkpint_index]
    bash drs2tuple.sh [start_checkpoint_index] [end_checkpint_index]
    
    python D-match/d-match.py -f1 dev.tuple.align_drs.gold -f2 [checkpoint_index].tuple -pr -r 100 -p 10
    
We choose the model with the highest F1 on develpment dataset as the final model, and trained model can be download from https://drive.google.com/open?id=1rzr4nd67tGHNo6T099e_FDxZkBVRFwbB.

## Easy-use
Download the pretrained model and move it to director workspace/gd_sys1

    cd TreeDRSparsing/workspace/gd_sys1
    tar -xvf models.tar
    bash easy.sh sample


