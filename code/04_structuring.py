import os
import torch
import transformers
import numpy as np
import json


def create_prompt(input, assistant_prompt):
    user_prompt = {"role": "user", "content": "\n".join(input)}

    return [user_prompt, assistant_prompt]


def list_in_json(string_list):
    string_list = string_list.strip()
    entry_list = string_list.strip('[').strip(']').split(';')

    data_dict = {'Nummer': None, 'Nachname': None, 'Vorname': None, 'Titel': None, 'Beruf': None,
                 'Sozialer Stand': None,
                 'Begleitung': None, 'Wohnort': None, 'Wohnung': None, 'Personenanzahl': None}
    for entry in entry_list:
        try:
            key, value = entry.split(":=")
        except:
            continue
        key = key.lstrip()
        value = value.lstrip()

        if key in data_dict.keys():
            if value == "null":
                data_dict[key] = None
            else:
                data_dict[key] = value

    data_json_string = json.dumps(data_dict, ensure_ascii=False)
    data_json = json.loads(data_json_string)
    return data_json


def merge_json_lists(*json_lists):
    merged_list = []

    for json_list in json_lists:

        if isinstance(json_list, str):
            try:
                json_list = json.loads(json_list)
            except json.JSONDecodeError:
                print(f"No valid JSON-string: {json_list}")
                continue

        if isinstance(json_list, list):
            merged_list.extend(json_list)
        else:
            # print(f"No valid list: {json_list}")
            try:
                merged_list.extend([json_list])
            except:
                print(f"Adding brackets did not help")

    return merged_list


#os.environ["CUDA_VISIBLE_DEVICES"]="7"

# model
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# model quantization
quantize = True

# huggingface token must be generated at https://huggingface.co/, you need to create an account
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

# structuring for all files
datapath = '../data/004_OCR_correction_all_tables'

years = os.listdir(datapath)

assistant_prompt = {"role": "assistant", "content": ""}

if not os.path.exists('../data/005_Structuring_all_tables'):
    os.makedirs('../data/005_Structuring_all_tables')

for year in years:

    system_prompt = {"role": "system", "content": "You are an expert in structuring data from tables. \n\
    Always include all of the following keys into the structuring: Nummer, Nachname, Vorname, Titel, Beruf, Sozialer Stand, Begleitung, Wohnort, Wohnung und Personenanzahl. \n\
    The output for the line '1000 Peter Fox ...' shall look like: [Nummer:=1000;Nachname:=Fox;Vorname:=Peter;...] \n\
    You always get a single line and only output one element. \n\
    If no data is available for a key, fill the value with null without quotation marks. \n\
    All words have to be included and no word must be removed. \n\
    The data origins from a row of a table that was extracted from a magazine between 1910 and 1932. \n\
    Do not output any comments or explanations! \n\
    Here are some additional rules that apply to the data: \n\
    Do not replace or encode german umlauts or 'ß'. \n\
    Always decode UTF-8 symbols. \n\
    Nummer is a large number and always the first element if it is contained. Otherwise it is null. Do not mix up with Personenanzahl.\n\
    Sometimes Vorname is just abbreviated. \n\
    Dr. belongs to Nachname. \n\
    Kind belongs to Sozialer Stand."}

    system_prompt_tokens = tokenizer.apply_chat_template([system_prompt], add_generation_prompt=False,
                                                         return_tensors="pt").to("cuda")

    file_list = [tmp for tmp in os.listdir(os.path.join(datapath, year)) if not tmp.endswith('.txt')]
    for file_name in file_list:
        tables = list(
            set(["".join(tmp.split("_", 2)[-1]) for tmp in os.listdir(os.path.join(datapath, year, file_name)) if
                 not tmp.startswith('.')]))
        for table in tables:

            os.makedirs(os.path.join('../data/005_Structuring_all_tables', year, file_name,
                                     'structuring_OCR_corrected_data_' + table.strip('.jpg')), exist_ok=True)

            if os.path.exists(os.path.join('../data/005_Structuring_all_tables', year, file_name,
                                           'structuring_OCR_corrected_data_' + table.strip('.jpg'),
                                           'structured_elements.json')):
                continue

            with open(
                    os.path.join(datapath, year, file_name, "ocr_correction_" + table.replace('.jpg', ''), 'correction',
                                 "ocr_correct.txt"), 'r') as f:
                full_corrected_ocr_data = f.read()

            # prepare ocr data -> merge two liner
            full_corrected_ocr_data_lines_reverse = full_corrected_ocr_data.splitlines()[::-1]
            threshold = np.median([len(x) for x in full_corrected_ocr_data_lines_reverse]) / 2

            merged_lines = []
            storage = ''
            for line in full_corrected_ocr_data_lines_reverse:
                if len(line) < threshold:
                    storage = line
                else:
                    if not storage == "":
                        tmp = line + ' ' + storage
                        storage = ''
                    else:
                        tmp = line
                    merged_lines.append(tmp)
            full_corrected_ocr_data_merged_lines = merged_lines[::-1]

            print('start structuring: ' + year + ' ' + file_name + ' ' + table)
            chunk_size = 1
            chunk_list = [np.array(full_corrected_ocr_data_merged_lines)[i:i + chunk_size] for i in
                          range(0, len(np.array(full_corrected_ocr_data_merged_lines)), chunk_size)]
            response_list = [[] for _ in range(len(chunk_list))]
            for idx, chunk in enumerate(chunk_list):
                my_prompt = create_prompt(chunk, assistant_prompt)
                try:
                    input_tokens = tokenizer.apply_chat_template(my_prompt, add_generation_prompt=False,
                                                                 return_tensors="pt").to("cuda")
                    model_inputs = torch.cat([system_prompt_tokens, input_tokens], dim=-1)

                    generate_kwargs = {
                        "input_ids": model_inputs,
                        "max_length": 600,  # Adjust for total length
                        "do_sample": True,  # Use sampling for non-zero temperature (randomness)
                        "temperature": .2,
                        "eos_token_id": terminators,  # Specify tokens to stop generation
                        'pad_token_id': tokenizer.eos_token_id,
                    }

                    generated_ids = model_4bit.generate(**generate_kwargs)
                    response = tokenizer.decode(generated_ids[0][model_inputs.shape[1]:], skip_special_tokens=True)
                    response_list[idx] = response
                except:
                    pass

            # merge all lists and convert to json
            json_list = [[] for _ in range(len(response_list))]
            for idx, result_chunk in enumerate(response_list):
                result_chunk = result_chunk.replace('assistant\n\n', '')
                tmp = []
                for entry in result_chunk.splitlines():
                    if not entry == '':
                        json_entry = list_in_json(entry)
                        json_list[idx].append(json_entry)

            prediction = merge_json_lists(*json_list)

            with open(os.path.join('../data/005_Structuring_all_tables', year, file_name,
                                   'structuring_OCR_corrected_data_' + table.strip('.jpg'), 'structured_elements.json'),
                      'w') as f:
                json.dump(prediction, f, ensure_ascii=False, indent=4)