
def specialize_tokenizer(tokenizer, templateFile = "template.jinja"):
    with open(templateFile) as templateFile:
        template = templateFile.read().strip()
        tokenizer.chat_template = template

    x = lambda i: ('<|' + i + '_start|>', '<|' + i + '_end|>')
    special_tokens = (x(i) for i in ['loss']) # , 'system', 'think', 'tool_call', 'tool_desc', 'tool_result' ])

    special_tokens = {
            'additional_special_tokens': [
                t 
                for tokens in special_tokens
                for t in tokens
        ]}

    tokenizer.pad_token = tokenizer.eos_token
    tokens_added = tokenizer.add_special_tokens(special_tokens)
    return tokens_added 
            
