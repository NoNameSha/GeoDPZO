#!/bin/bash

# Main settings with default values
TASK=${TASK:-"SST-2"}           # see all the options in the "cases" below
SEED=${SEED:-13}                # random seed and also data seed, by default the data split seeds are {13, 21, 42, 87, 100}
K=${K:-16}                      # choose from {16, 64, 512} by default
MODEL=${MODEL:-"roberta-large"}  # pick a RoBERTa or BERT model - roberta-large
TYPE=${TYPE:-"prompt"}          # fine-tuning setting, choose from "finetune" and "prompt"
TRAINER=${TRAINER:-"standard"}
TAG=${TAG:-}                    # set a tag to distinguish and aggregate runs in the log
NUM_GPU=${NUM_GPU:-1}           # by default use 1 GPU, set to 0 for CPU-only training
OPT=${OPT:-"adam"}
STEPS=${STEPS:-800}

TASK_EXTRA=""
case $TASK in
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        ;; 
    sst-5)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --double_demo"
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    SNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    trec)
        TEMPLATE="*cls**mask*:*+sent_0**sep+*"
        MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    mr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    cr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    mpqa)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    CoLA)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{'0':'incorrect','1':'correct'}"
        ;;
    subj)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{0:'subjective',1:'objective'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    RTE)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    go_emotions)
        TEMPLATE=*cls**sent_0*_The_emotion_expressed_was*mask*.*sep+*
        #MAPPING="{'0':'admiration','1':'amusement','2':'anger','3':'annoyed','4':'approval','5':'caring','6':'confused','7':'curious','8':'desire','9':'disappointed','10':'disapproval','11':'disgust','12':'embarrassed','13':'excited','14':'fear','15':'grateful','16':'grief','17':'joy','18':'love','19':'nervous','20':'optimistic','21':'proud','22':'realized','23':'relieved','24':'guilty','25':'sad','26':'surprised','27':'neutral'}"
        MAPPING="{0:'admiration',1:'amusement',2:'anger',3:'annoyed',4:'approval',5:'caring',6:'confused',7:'curious',8:'desire',9:'disappointed',10:'disapproval',11:'disgust',12:'embarrassed',13:'excited',14:'fear',15:'grateful',16:'grief',17:'joy',18:'love',19:'nervous',20:'optimistic',21:'proud',22:'realized',23:'relieved',24:'guilty',25:'sad',26:'surprised',27:'neutral'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --double_demo"
        ;;
    yahoo_answers) 
        TEMPLATE=*cls**sent_0*_This_question_is_about*mask*.*sep+* 
        MAPPING="{0:'Society',1:'Science',2:'Health',3:'Education',4:'Tech',5:'Sports',6:'Business',7:'Movies',8:'Family',9:'Politics'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20" 
        ;;
        # TEMPLATE="*sent_0*_This_question_is_about"
        # MAPPING="{0:'Society',1:'Science',2:'Health',3:'Education',4:'Tech',5:'Sports',6:'Business',7:'Movies',8:'Family',9:'Politics'}"
        # TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20"
        # ;;
esac

ALL_ARGS_TOGETHER="
    --model_name_or_path $MODEL --few_shot_type $TYPE
    --task_name $TASK --template $TEMPLATE --mapping $MAPPING
    --data_dir data/k-shot-1k-test/$TASK/$K-$SEED
    --overwrite_output_dir --output_dir result/$TASK-$MODEL-$TYPE-$TRAINER-$TAG$GRID_TAG/$K-$SEED
    --num_k $K
    --tag $TAG
    --max_seq_length 128
    --seed $SEED
    --do_eval --do_predict --do_train
    --trainer $TRAINER
    --optimizer $OPT --max_steps $STEPS
    --logging_steps 10
    --per_device_eval_batch_size 4
    $TASK_EXTRA
    $LOAD_KERNELS
    $@
"

if [[ $NUM_GPU > 1 ]]; then
    # Randomly set a port number
    # If you encounter "address already used" error, just run again or manually set an available port id.
    PORT_ID=$(expr $RANDOM + 1000)

    # Allow multiple threads
    export OMP_NUM_THREADS=8

    python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID run_aggzo.py \
        $ALL_ARGS_TOGETHER
else
    python run_aggzo.py \
        $ALL_ARGS_TOGETHER
fi

rm -rf result/$TASK-$MODEL-$TYPE-$TRAINER-$TAG$GRID_TAG/$K-$SEED