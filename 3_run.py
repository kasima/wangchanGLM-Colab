# @title Run
# @markdown วิธีการใช้งาน สำหรับแชทให้คลิกแท็บ chatbot ส่วนหากต้องการสร้างข้อความให้ไปที่ Text Generation
import copy
from urllib.request import Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import requests
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

template = """
{history}
<human>: {human_input}
<bot>:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template
)
exclude_pattern = re.compile(r'[^ก-๙]+')  # |[^0-9a-zA-Z]+


def is_exclude(text):
   return bool(exclude_pattern.search(text))


df = pd.DataFrame(tokenizer.vocab.items(), columns=['text', 'idx'])
df['is_exclude'] = df.text.map(is_exclude)
exclude_ids = df[df.is_exclude == True].idx.tolist()
if Thai == "Yes":
  pipe = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      max_new_tokens=512,
      begin_suppress_tokens=exclude_ids,
      no_repeat_ngram_size=2,
  )
else:
  pipe = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      max_new_tokens=512,
      no_repeat_ngram_size=2,
  )
hf_pipeline = HuggingFacePipeline(pipeline=pipe)

chatgpt_chain = LLMChain(
    llm=hf_pipeline,
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)


api_url = "https://wangchanglm.numfa.com/api.php"  # Don't open this url!!!


def sumbit_data(save, prompt, vote, feedback=None, max_len=None, temp=None, top_p=None, name_model=name_model):
  api_url = "https://wangchanglm.numfa.com/api.php"
  myobj = {
      'save': save,
      'prompt': prompt,
      'vote': vote,
      'feedback': feedback,
      'max_len': max_len,
      'temp': temp,
      'top_p': top_p,
      'model': name_model
  }
  _temp_url = "https://wangchanglm.numfa.com/api.php"
  _temp_url += "?" + urlencode(myobj, doseq=True, safe="/")
  html = urlopen(_temp_url).read().decode('utf-8')
  return True


def gen_instruct(text, max_new_tokens=512, top_p=0.95, temperature=0.9, top_k=50):
    batch = tokenizer(text, return_tensors="pt")
    with torch.cuda.amp.autocast():  # cuda -> cpu if cpu
        if Thai == "Yes":
          output_tokens = model.generate(
              input_ids=batch["input_ids"],
              max_new_tokens=max_new_tokens,  # 512
              begin_suppress_tokens=exclude_ids,
              no_repeat_ngram_size=2,
              # oasst k50
              top_k=top_k,
              top_p=top_p,  # 0.95
              typical_p=1.,
              temperature=temperature,  # 0.9
          )
        else:
          output_tokens = model.generate(
              input_ids=batch["input_ids"],
              max_new_tokens=max_new_tokens,  # 512
              no_repeat_ngram_size=2,
              # oasst k50
              top_k=top_k,
              top_p=top_p,  # 0.95
              typical_p=1.,
              temperature=temperature,  # 0.9
          )
    return tokenizer.decode(output_tokens[0][len(batch["input_ids"][0]):], skip_special_tokens=True)


def gen_chatbot_old(text):
    is_sensitive, respond_message = guardian.filter(text)
    if is_sensitive:
        return respond_message

    batch = tokenizer(text, return_tensors="pt")
    # context_tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    # logits_processor = FocusContextProcessor(context_tokens, model.config.vocab_size, scaling_factor = 1.5)
    with torch.cpu.amp.autocast():  # cuda if gpu
        output_tokens = model.generate(
            input_ids=batch["input_ids"],
            max_new_tokens=512,
            begin_suppress_tokens=exclude_ids,
            no_repeat_ngram_size=2,
        )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True).split(": ")[-1]


def list2prompt(history):
    _text = ""
    for user, bot in history:
        _text += "<human>: "+user+"\n<bot>: "
        if bot != None:
            _text += bot+"\n"
    return _text


PROMPT_DICT = {
    "prompt_input": (
        "<context>: {input}\n<human>: {instruction}\n<bot>: "
    ),
    "prompt_no_input": (
        "<human>: {instruction}\n<bot>: "
    ),
}


def instruct_generate(
    instruct: str,
    input: str = 'none',
    max_gen_len=512,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    is_sensitive, respond_message = guardian.filter(instruct)
    if is_sensitive:
        return respond_message

    if input == 'none' or len(input) < 2:
        prompt = PROMPT_DICT['prompt_no_input'].format_map(
            {'instruction': instruct, 'input': ''})
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(
            {'instruction': instruct, 'input': input})
    result = gen_instruct(prompt, max_gen_len, top_p, temperature)
    return result


with gr.Blocks(height=900) as demo:
    chatgpt_chain = LLMChain(
        llm=hf_pipeline,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2),
    )
    with gr.Tab("Text Generation"):
        with gr.Row():
            with gr.Column():
                instruction = gr.Textbox(
                    lines=2, label="Instruction", max_lines=10)
                input = gr.Textbox(
                    lines=2, label="Context input", placeholder='none', max_lines=5)
                max_len = gr.Slider(minimum=1, maximum=1024,
                                    value=512, label="Max new tokens")
                with gr.Accordion(label='Advanced options', open=False):
                    temp = gr.Slider(minimum=0, maximum=1,
                                     value=0.9, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1,
                                      value=0.95, label="Top p")

                run_botton = gr.Button("Run")

            with gr.Column():
                outputs = gr.Textbox(lines=10, label="Output")
                with gr.Column(visible=False) as feedback_gen_box:
                    gen_radio = gr.Radio(
                        ["Good", "Bad", "Report"], label="Do you think about the chat?")
                    feedback_gen = gr.Textbox(
                        placeholder="Feedback chatbot", show_label=False, lines=4)
                    feedback_gen_submit = gr.Button("Submit Feedback")
                with gr.Row(visible=False) as feedback_gen_ok:
                    gr.Markdown("Thank you for feedback.")

        def save_up2(instruction, input, prompt, max_len, temp, top_p, choice, feedback):
            save = "gen"
            if input == 'none' or len(input) < 2:
              _prompt = PROMPT_DICT['prompt_no_input'].format_map(
                  {'instruction': instruction, 'input': ''})
            else:
              _prompt = PROMPT_DICT['prompt_input'].format_map(
                  {'instruction': instruction, 'input': input})
            prompt = _prompt+prompt
            if choice == "Good":
              sumbit_data(save=save, prompt=prompt, vote=1,
                          feedback=feedback, max_len=max_len, temp=temp, top_p=top_p)
            elif choice == "Bad":
              sumbit_data(save=save, prompt=prompt, vote=0,
                          feedback=feedback, max_len=max_len, temp=temp, top_p=top_p)
            else:
              sumbit_data(save=save, prompt=prompt, vote=3,
                          feedback=feedback, max_len=max_len, temp=temp, top_p=top_p)
            return {feedback_gen_box: gr.update(visible=False), feedback_gen_ok: gr.update(visible=True)}

        def gen(instruct: str, input: str = 'none', max_gen_len=512, temperature: float = 0.1, top_p: float = 0.75):
            feedback_gen_ok.update(visible=False)
            _temp = instruct_generate(
                instruct, input, max_gen_len, temperature, top_p)
            feedback_gen_box.update(visible=True)
            return {outputs: _temp, feedback_gen_box: gr.update(visible=True), feedback_gen_ok: gr.update(visible=False)}
        feedback_gen_submit.click(fn=save_up2, inputs=[instruction, input, outputs, max_len, temp, top_p, gen_radio, feedback_gen], outputs=[
                                  feedback_gen_box, feedback_gen_ok], queue=True)
        inputs = [instruction, input, max_len, temp, top_p]
        run_botton.click(fn=gen, inputs=inputs, outputs=[
                         outputs, feedback_gen_box, feedback_gen_ok])
        examples = gr.Examples(examples=["แต่งกลอนวันแม่", "แต่งกลอนแปดวันแม่", 'อยากลดความอ้วนทำไง',
                               'จงแต่งเรียงความเรื่องความฝันของคนรุ่นใหม่ต่อประเทศไทย'], inputs=[instruction])
    with gr.Tab("ChatBot"):
        with gr.Column():
            chatbot = gr.Chatbot(
                label="Chat Message Box", placeholder="Chat Message Box", show_label=False).style(container=False)
        with gr.Row():
          with gr.Column(scale=0.85):
            msg = gr.Textbox(
                placeholder="พิมพ์คำถามของคุณที่นี่... (กด enter หรือ submit หลังพิมพ์เสร็จ)", show_label=False)
          with gr.Column(scale=0.15, min_width=0):
            submit = gr.Button("Submit")
        with gr.Column():
            with gr.Column(visible=False) as feedback_chatbot_box:
                chatbot_radio = gr.Radio(
                    ["Good", "Bad", "Report"], label="Do you think about the chat?"
                )
                feedback_chatbot = gr.Textbox(
                    placeholder="Feedback chatbot", show_label=False, lines=4)
                feedback_chatbot_submit = gr.Button("Submit Feedback")
            with gr.Row(visible=False) as feedback_chatbot_ok:
                gr.Markdown("Thank you for feedback.")
        clear = gr.Button("Clear")

        def save_up(history, choice, feedback):
            _bot = list2prompt(history)
            x = False
            if choice == "Good":
              x = sumbit_data(save="chat", prompt=_bot,
                              vote=1, feedback=feedback)
            elif choice == "Bad":
              x = sumbit_data(save="chat", prompt=_bot,
                              vote=0, feedback=feedback)
            else:
              x = sumbit_data(save="chat", prompt=_bot,
                              vote=3, feedback=feedback)
            return {feedback_chatbot_ok: gr.update(visible=True), feedback_chatbot_box: gr.update(visible=False)}

        def user(user_message, history):
            is_sensitive, respond_message = guardian.filter(user_message)
            if is_sensitive:
                bot_message = respond_message
            else:
                bot_message = chatgpt_chain.predict(human_input=user_message)
            history.append((user_message, bot_message))
            return "", history, gr.update(visible=True)

        def reset():
          chatgpt_chain.memory.clear()
          print("clear!")
        feedback_chatbot_submit.click(fn=save_up, inputs=[chatbot, chatbot_radio, feedback_chatbot], outputs=[
                                      feedback_chatbot_ok, feedback_chatbot_box,], queue=True)
        clear.click(reset, None, chatbot, queue=False)
        submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[
                                  msg, chatbot, feedback_chatbot_box], queue=True)
        submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[
                                          msg, chatbot, feedback_chatbot_box], queue=True)
    with gr.Tab("ChatBot without LangChain"):
        chatbot2 = gr.Chatbot()
        msg2 = gr.Textbox(
            label="Your sentence here... (press enter to submit)")
        with gr.Column():
            with gr.Column(visible=False) as feedback_chatbot_box2:
                chatbot_radio2 = gr.Radio(
                    ["Good", "Bad", "Report"], label="Do you think about the chat?"
                )
                feedback_chatbot2 = gr.Textbox(
                    placeholder="Feedback chatbot", show_label=False, lines=4)
                feedback_chatbot_submit2 = gr.Button("Submit Feedback")
            with gr.Row(visible=False) as feedback_chatbot_ok2:
                gr.Markdown("Thank you for feedback.")

        def user2(user_message, history):
            return "", history + [[user_message, None]]

        def bot2(history):
            _bot = list2prompt(history)
            bot_message = gen_chatbot_old(_bot)
            history[-1][1] = bot_message
            return history, gr.update(visible=True)

        def save_up2(history, choice, feedback):
            _bot = list2prompt(history)
            x = False
            if choice == "Good":
              x = sumbit_data(save="chat", prompt=_bot, vote=1,
                              feedback=feedback, name_model=name_model+"-chat_old")
            elif choice == "Bad":
              x = sumbit_data(save="chat", prompt=_bot, vote=0,
                              feedback=feedback, name_model=name_model+"-chat_old")
            else:
              x = sumbit_data(save="chat", prompt=_bot, vote=3,
                              feedback=feedback, name_model=name_model+"-chat_old")
            return {feedback_chatbot_ok2: gr.update(visible=True), feedback_chatbot_box2: gr.update(visible=False)}
        msg2.submit(user2, [msg2, chatbot2], [msg2, chatbot2], queue=False).then(
            bot2, chatbot2, [chatbot2, feedback_chatbot_box2])
        feedback_chatbot_submit2.click(fn=save_up2, inputs=[chatbot2, chatbot_radio2, feedback_chatbot2], outputs=[
                                       feedback_chatbot_ok2, feedback_chatbot_box2], queue=True)
        clear2 = gr.Button("Clear")
        clear2.click(lambda: None, None, chatbot2, queue=False)
demo.queue()
demo.launch()
