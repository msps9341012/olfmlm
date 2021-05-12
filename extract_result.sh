#!/bin/bash

#res=$(grep "$1" /iesl/canvas/rueiyaosun/finetuned_berts/mf+mlm/results.tsv)




# Getmax () {
#    F=$(echo $res | grep -oE "$1:[0-9 .]+")
#    number=$(echo $F | grep -oE "\s[0-9 .]+" )
#    #echo $number
#    myarray=($number)
#    max=$(printf '%s\n' "${myarray[@]}" | awk '$1 > m || NR == 1 { m = $1 } END { print m }')
#    echo "$1: $max"
# }
# Getmax mnli_accuracy
# Getmax qqp_f1
# Getmax qnli_accuracy
# Getmax sst_accuracy
# Getmax cola_mcc
# Getmax sts-b_spearmanr
# Getmax mrpc_f1
# Getmax rte_accuracy

Getall(){
    res=$(grep -E -- "^$1.*$2" /iesl/canvas/rueiyaosun/finetuned_berts/mf+mlm/results.tsv)
    marco_number=$(echo $res |grep -oE "macro_avg:[0-9 .]+" | grep -oE "\s[0-9 .]+")
    myarray=($marco_number)
    marco_max=$(printf '%s\n' "${myarray[@]}" | awk '$1 > m || NR == 1 { m = $1 } END { print m }')
    
    metric="$2_$3"
    m=$(echo $res |grep -oE "$metric:[0-9 .]+" | grep -oE "\s[0-9 .]+")
    myarray=($m)
    m_max=$(printf '%s\n' "${myarray[@]}" | awk '$1 > m || NR == 1 { m = $1 } END { print m }')
    results=($m_max $marco_max)
    echo ${results[@]}
}

r=($(Getall $1 mnli accuracy))
echo "mnli_accuracy: ${r[0]}"
total=`echo 0 + ${r[1]} | bc`

r=($(Getall $1 qqp f1))
echo "qqp_f1: ${r[0]}"
total=`echo $total + ${r[1]} | bc`

r=($(Getall $1 qnli accuracy))
echo "qnli_accuracy: ${r[0]}"
total=`echo $total + ${r[1]} | bc`


r=($(Getall $1 sst accuracy))
echo "sst_accuracy: ${r[0]}"
total=`echo $total + ${r[1]} | bc`

r=($(Getall $1 cola mcc))
echo "cola_mcc: ${r[0]}"
total=`echo $total + ${r[1]} | bc`

r=($(Getall $1 sts-b spearmanr))
echo "sts-b_spearmanr: ${r[0]}"
total=`echo $total + ${r[1]} | bc`

r=($(Getall $1 mrpc f1))
echo "mrpc_f1: ${r[0]}"
total=`echo $total + ${r[1]} | bc`

r=($(Getall $1 rte accuracy))
echo "rte_accuracy: ${r[0]}"
total=`echo $total + ${r[1]} | bc`

calc(){ awk "BEGIN { print "$*" }"; }
avg=$(calc $total/8)
echo "glue score: $avg"







# Getmarco(){
#     F=$(grep -E -- "$1.*$2" finetuned_berts/rg+mlm/results.tsv | grep -oE "macro_avg:[0-9 .]+")
#     number=$(echo $F | grep -oE "\s[0-9 .]+" )
#     myarray=($number)
#     max=$(printf '%s\n' "${myarray[@]}" | awk '$1 > m || NR == 1 { m = $1 } END { print m }')
# }

# $cola=Getmarco $1 cola 
# $cola=Getmarco $1 cola 




# F=$(echo $res | grep -oE "mnli_accuracy:[0-9 .]+")
# number=$(echo $F | sed 's/[^ .0-9]*//g' )
# myarray=($number)
# IFS=$'\n'
# max=$(echo "${myarray[*]}" | sort -nr | head -n1)
# echo "mnli_accuracy: $max"




#echo $res | grep -oE "mnli_accuracy:[0-9 .]+"
#echo $res | grep -oE "qqp_f1:[0-9 .]+"
#echo $res | grep -oE "qnli_accuracy:[0-9 .]+"
#echo $res | grep -oE "sst_accuracy:[0-9 .]+"
#echo $res | grep -oE "cola_mcc:[0-9 .]+"
#echo $res | grep -oE "sts-b_spearmanr:[0-9 .]+"
#echo $res | grep -oE "mrpc_f1:[0-9 .]+"
#echo $res | grep -oE "rte_accuracy:[0-9 .]+"








