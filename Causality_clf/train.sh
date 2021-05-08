stdbuf -oL python3 -u main.py  | while IFS= read -r line
do
tee -a model_log.txt
done
