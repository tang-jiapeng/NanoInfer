python export_llama2.py  ../models/llama2/llama2_fp32.bin --hf ../models/llama2  --version 0 --dtype fp32

python export_llama3.py  ../models/llama3/llama3_fp32.bin --hf ../models/llama3  --version 0 --dtype fp32

hf download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama3/ 