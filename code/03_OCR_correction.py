import torch
import transformers
import os
import numpy as np
import re
import sys

sys.path.insert(1, 'util')
from alignment import align_blocks

# os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5,6,7"

# model
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# model quantization
quantize = True

# huggingface token must be generated at https://huggingface.co/ after creating an account
token_file = "../huggingface_token.txt"

try:
    with open(token_file, 'r') as f:
        api_access_token = f.read().strip()
except FileNotFoundError:
    raise FileNotFoundError(f"{token_file} file not found!")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=api_access_token)
terminators = [
    #tokenizer.eos_token_id,  # End-of-sentence token
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Custom end-of-conversation token
]

model_4bit = transformers.AutoModelForCausalLM.from_pretrained(model_id,
                                                   token=api_access_token,
                                                   torch_dtype=torch.bfloat16,
                                                   cache_dir="./",
                                                   quantization_config=transformers.BitsAndBytesConfig(load_in_4bit=quantize),
                                                   device_map="auto",
                                                    eos_token_id=terminators)

# OCR correction for processing all files
def create_prompt(input, system_prompt, assistent_prompt):
    user_prompt = {"role": "user", "content": "\n".join(input)}

    return [system_prompt, user_prompt, assistant_prompt]


datapath = '../data/002_OCR_all_tables'
years = os.listdir(datapath)

system_prompt = {"role": "system", "content": "You are an expert for OCR and in correcting small errors. \n\
   You are given three rows of a German table that were created by three different OCR systems. \n\
   Your task is to create the correct output based all entries. \n\
   Just output the result and no explainations. \n\
   Don't replace the German letter 'ß' by 'ss' \n\
   Please keep ALL vertical bars. \n\
   Please keep ALL ditto marks. \n\
   Some whitespaces might seperate words belonging together. \n\
   Please replace multiple sequential whitespaces by one. \n\
   Create the correct output. \n\
   In the following you are provided a tuple of such rows in parentheses. The rows are separated by ';' resulting in '(ocr_row1;ocr_row2;ocr_row3)' \n\
   A row can contain an empty string. In this case the tuple looks like '(;ocr_row2;ocr_row3)','(ocr_row1;;ocr_row3)' or '(ocr_row1;ocr_row2;)'. Hence perform the correction with two ocr_rows only. \n\
   If one ocr_row does not provide any useful information, it can be omitted for correction. \n\
   For each pair, create the correct table row and output all of them line by line. \n\
   Skip table headers from both input rows and do not include empty lines. \n\
   "}

assistant_prompt = {"role": "assistant", "content": ""}

if not os.path.exists('../data/004_OCR_correction_all_tables'):
    os.makedirs('../data/004_OCR_correction_all_tables')

for year in years:

    if not os.path.exists(os.path.join('../data/004_OCR_correction_all_tables', year)):
        os.makedirs(os.path.join('../data/004_OCR_correction_all_tables', year))

    file_list = [tmp for tmp in os.listdir(os.path.join(datapath, year)) if not tmp.endswith('.txt')]
    for file_name in file_list:

        if not os.path.exists(os.path.join('../data/004_OCR_correction_all_tables', year, file_name)):
            os.makedirs(os.path.join('../data/004_OCR_correction_all_tables', year, file_name))

        tables = list(
            set(["_".join(tmp.split("_", 2)[:2]) for tmp in os.listdir(os.path.join(datapath, year, file_name)) if
                 tmp.endswith('txt')]))
        for table in tables:
            ocr_files = [tmp for tmp in os.listdir(os.path.join(datapath, year, file_name)) if
                         tmp.endswith('.txt') and table in tmp]

            os.makedirs(os.path.join('../data/004_OCR_correction_all_tables', year, file_name, 'ocr_correction_' + table),
                        exist_ok=True)
            os.makedirs(os.path.join('../data/004_OCR_correction_all_tables', year, file_name, 'ocr_correction_' + table,
                                     'correction'), exist_ok=True)

            full_ocr_data = [[], [], []]
            for ocr_idx, ocr_file in enumerate(ocr_files):

                with open(os.path.join(datapath, year, file_name, ocr_file), 'r') as f:
                    full_ocr_data_tmp = f.read()

                if 'frak2021' in ocr_file:
                    full_ocr_data[0] = full_ocr_data_tmp
                if 'Fraktur' in ocr_file:
                    full_ocr_data[1] = full_ocr_data_tmp
                if 'GT4HistOCR' in ocr_file:
                    full_ocr_data[2] = full_ocr_data_tmp

            aligned_ocr_file1, aligned_ocr_file2 = align_blocks(full_ocr_data[0].splitlines(),
                                                                full_ocr_data[1].splitlines(), None, 30)
            aligned_ocr_file1, aligned_ocr_file3 = align_blocks(aligned_ocr_file1, full_ocr_data[2].splitlines(), None,
                                                                30)
            aligned_ocr_file1, aligned_ocr_file2 = align_blocks(aligned_ocr_file1, aligned_ocr_file2, None, 30)

            aligned_ocr_data = [[], [], []]  # pos 0: frak2021, pos 1: Fraktur, pos 2: GT4HistOCR
            aligned_ocr_data[0] = aligned_ocr_file1
            aligned_ocr_data[1] = aligned_ocr_file2
            aligned_ocr_data[2] = aligned_ocr_file3

            # store aligned data
            with open(os.path.join('../data/004_OCR_correction_all_tables', year, file_name,
                                   'ocr_correction_' + table.strip('.jpg'), 'aligned_frak2021_data.txt'), 'w') as f:
                f.write("\n".join(aligned_ocr_data[0]))
            with open(os.path.join('../data/004_OCR_correction_all_tables', year, file_name,
                                   'ocr_correction_' + table.strip('.jpg'), 'aligned_Fraktur_data.txt'), 'w') as f:
                f.write("\n".join(aligned_ocr_data[1]))
            with open(os.path.join('../data/004_OCR_correction_all_tables', year, file_name,
                                   'ocr_correction_' + table.strip('.jpg'), 'aligned_GT4HistOCR_data.txt'), 'w') as f:
                f.write("\n".join(aligned_ocr_data[2]))

            # prepare data -> use order 213 from evaluation -> Fraktur,frak2021,GT4HistOCR
            ocr_pairs = []
            for line1, line2, line3 in zip(aligned_ocr_data[0], aligned_ocr_data[1], aligned_ocr_data[2]):
                line1 = line1.replace('\n','')
                line2 = line2.replace('\n','')
                line3 = line3.replace('\n','')
                ocr_pairs.append(f"({line1};{line2};{line3})")

            print('start correcting: ' + year + ' ' + file_name + ' ' + table)
            chunk_size = 1
            chunk_list = [np.array(ocr_pairs)[i:i + chunk_size] for i in range(0, len(np.array(ocr_pairs)), chunk_size)]
            result = ''
            for chunk_idx, chunk in enumerate(chunk_list):
                my_prompt = create_prompt(chunk, system_prompt, assistant_prompt)

                model_inputs = tokenizer.apply_chat_template(my_prompt, add_generation_prompt=False,
                                                             return_tensors="pt").to("cuda")

                generate_kwargs = {
                    "input_ids": model_inputs,
                    "max_length": 10000,  # Adjust for total length
                    "do_sample": True,  # Use sampling for non-zero temperature (randomness)
                    "temperature": .2,
                    "eos_token_id": terminators,  # Specify tokens to stop generation
                    'pad_token_id': tokenizer.eos_token_id,
                }

                generated_ids = model_4bit.generate(**generate_kwargs)

                tmp = tokenizer.decode(generated_ids[0][model_inputs.shape[1]:], skip_special_tokens=True)
                tmp = "".join([s for s in tmp.splitlines(True) if s.strip()])  # remove blank lines
                tmp = re.sub(' +', ' ', tmp)  # remove multiple white spaces
                if result == '':
                    result = tmp
                else:
                    result = result + '\n' + tmp
            response = result

            with open(os.path.join('../data/004_OCR_correction_all_tables', year, file_name,
                                   'ocr_correction_' + table.strip('.jpg'), 'correction', 'ocr_correct.txt'), 'w') as f:
                f.write(response)
