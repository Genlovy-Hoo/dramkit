# 添加key到环境变量
export OPENAI_API_KEY="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"

# 生成jsonl格式文件
openai tools fine_tunes.prepare_data -f ./openai_finetuning_dataset.csv

# 上传文件创建微调任务
openai api fine_tunes.create -t ./openai_finetuning_dataset_prepared.jsonl -m davinci --suffix "hsgpt"
# 记录返回的两个id
# file-id: file-PI4JGI8lAJrYtq4giW45DYH7
# ft-id: ft-bCLtQwJQRkrloOpDoSPMhFbL
# 如果中断，重新查看：
openai api fine_tunes.follow -i ft-bCLtQwJQRkrloOpDoSPMhFbL

# 查看所有微调模型列表
openai api fine_tunes.list

# 查看指定微调模型任务
openai api fine_tunes.get -i ft-bCLtQwJQRkrloOpDoSPMhFbL

# 取消微调模型训练任务
openai api fine_tunes.cancel -i ft-bCLtQwJQRkrloOpDoSPMhFbL

# 记录result file id和model name
# file-BZrtEUWD09D6aoJMpmKxA4IK
# davinci:ft-personal:hsgpt-2023-03-15-10-10-47

# 测试模型
openai api completions.create -m davinci:ft-personal:hsgpt-2023-03-15-10-10-47 -p <YOUR_PROMPT>
