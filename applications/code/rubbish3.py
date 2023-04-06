from transformers import PLBartTokenizer, PLBartForSequenceClassification, PLBartConfig, PLBartForConditionalGeneration
import torch
# 加载预训练模型和分词器
model_name = "/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/ckpt_test/"#"/root/autodl-tmp/CodePrompt/data/huggingface_models/plbart-base/"#
model_type ='plbart'
ckpt= model_name #"/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1"

# func1='public String getData(DefaultHttpClient httpclient) { try { HttpGet get = new HttpGet("http://3dforandroid.appspot.com/api/v1/note"); get.setHeader("Content-Type", "application/json"); get.setHeader("Accept", "*/*"); HttpResponse response = httpclient.execute(get); HttpEntity entity = response.getEntity(); InputStream instream = entity.getContent(); responseMessage = read(instream); if (instream != null) instream.close(); } catch (ClientProtocolException e) { e.printStackTrace(); } catch (IOException e) { e.printStackTrace(); } return responseMessage; }'
# func2='public static void copyFile(File in, File out) throws Exception { FileChannel sourceChannel = new FileInputStream(in).getChannel(); FileChannel destinationChannel = new FileOutputStream(out).getChannel(); sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel); sourceChannel.close(); destinationChannel.close(); }'
# #label=0

# train by BigCloneBench dataset, so you can only do JAVA code clone detection
func1="""
public String getData(DefaultHttpClient httpclient) {
    try {
        HttpGet get = new HttpGet("http://3dforandroid.appspot.com/api/v1/note");
        get.setHeader("Content-Type", "application/json");
        get.setHeader("Accept", "*/*");
        HttpResponse response = httpclient.execute(get);
        HttpEntity entity = response.getEntity();
        InputStream instream = entity.getContent();
        responseMessage = read(instream);
        if (instream != null) instream.close();
    } catch (ClientProtocolException e) {
        e.printStackTrace();
    } catch (IOException e) {
        e.printStackTrace();
    }
    return responseMessage;
}
"""
#label=0
# func2='public static void copyFile(File in, File out) throws Exception { FileChannel sourceChannel = new FileInputStream(in).getChannel(); FileChannel destinationChannel = new FileOutputStream(out).getChannel(); sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel); sourceChannel.close(); destinationChannel.close(); }'
func2="""
public static void copyFile(File in, File out) throws Exception {
    FileChannel sourceChannel = new FileInputStream(in).getChannel();
    FileChannel destinationChannel = new FileOutputStream(out).getChannel();
    sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
    sourceChannel.close();
    destinationChannel.close();
}
"""
# #label=0

# func1='public static void copyFile(File source, File dest) throws IOException { FileChannel in = null, out = null; try { in = new FileInputStream(source).getChannel(); out = new FileOutputStream(dest).getChannel(); in.transferTo(0, in.size(), out); } catch (FileNotFoundException fnfe) { Log.debug(fnfe); } finally { if (in != null) in.close(); if (out != null) out.close(); } }'
# func2='public static void copyFile(File from, File to) throws IOException { if (from.isDirectory()) { if (!to.exists()) { to.mkdir(); } File[] children = from.listFiles(); for (int i = 0; i < children.length; i++) { if (children[i].getName().equals(".") || children[i].getName().equals("..")) { continue; } if (children[i].isDirectory()) { File f = new File(to, children[i].getName()); copyFile(children[i], f); } else { copyFile(children[i], to); } } } else if (from.isFile() && (to.isDirectory() || to.isFile())) { if (to.isDirectory()) { to = new File(to, from.getName()); } FileInputStream in = new FileInputStream(from); FileOutputStream out = new FileOutputStream(to); byte[] buf = new byte[32678]; int read; while ((read = in.read(buf)) > -1) { out.write(buf, 0, read); } closeStream(in); closeStream(out); } }'
# #label=1

# func1="""
# public static void copyFile(File source, File dest) throws IOException {
#     FileChannel in = null, out = null;
#     try {
#         in = new FileInputStream(source).getChannel();
#         out = new FileOutputStream(dest).getChannel();
#         in.transferTo(0, in.size(), out);
#     } catch (FileNotFoundException fnfe) {
#         Log.debug(fnfe);
#     } finally {
#         if (in != null) in.close();
#         if (out != null) out.close();
#     }
# }
# """

# func2="""
# public static void copyFile(File from, File to) throws IOException {
#     if (from.isDirectory()) {
#         if (!to.exists()) {
#             to.mkdir();
#         }
#         File[] children = from.listFiles();
#         for (int i = 0; i < children.length; i++) {
#             if (children[i].getName().equals(".") || children[i].getName().equals("..")) {
#                 continue;
#             }
#             if (children[i].isDirectory()) {
#                 File f = new File(to, children[i].getName());
#                 copyFile(children[i], f);
#             } else {
#                 copyFile(children[i], to);
#             }
#         }
#     } else if (from.isFile() && (to.isDirectory() || to.isFile())) {
#         if (to.isDirectory()) {
#             to = new File(to, from.getName());
#         }
#         FileInputStream in = new FileInputStream(from);
#         FileOutputStream out = new FileOutputStream(to);
#         byte[] buf = new byte[32678];
#         int read;
#         while ((read = in.read(buf)) > -1) {
#             out.write(buf, 0, read);
#         }
#         closeStream(in);
#         closeStream(out);
#     }
# }
# """


# func1="""
# def calculate_area(width, height):
#     return width * height

# rectangle_width = 4
# rectangle_height = 5
# area = calculate_area(rectangle_width, rectangle_height)
# print(f"矩形的面积为：{area}")
# """

# func2="""
# def calculate_area(width, height):
#     return width * height

# rectangle_width = 4
# rectangle_height = 5
# area = calculate_area(rectangle_width, rectangle_height)
# print(f"矩形的面积为：{area}")
# """


# """
# def compute_area(w, h):
#     return w * h

# rect_width = 4
# rect_height = 5
# rect_area = compute_area(rect_width, rect_height)
# print(f"矩形的面积为：{rect_area}")
# """


# func1 = """
# def factorial(n):
#     if n == 0:
#         return 1
#     else:
#         return n * factorial(n-1)
# """
# func2 = """
# def fact(n):
#     return 1 if n == 0 else n * fact(n - 1)
# """

import torch.nn as nn
import os
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CloneModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, model_type):
        super(CloneModel, self).__init__()
        # checkpoint = os.path.join(args.huggingface_locals, MODEL_LOCALS[args.model_name])
        # config = AutoConfig.from_pretrained(checkpoint)
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = ClassificationHead(config)
        self.model_type = model_type
        self.eos_token_id = 2


    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)

        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        # position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        # position_ids = position_ids*attention_mask
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        # attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        position_ids = position_ids*attention_mask
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def get_unixcoder_vec(self, source_ids):
        attention_mask = source_ids.ne(1)
        position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        position_ids = position_ids*attention_mask

        outputs = self.encoder(source_ids,attention_mask=attention_mask)[0]#shape:batch_size*max_len512*hidden_size768

        outputs = (outputs * source_ids.ne(1)[:,:,None]).sum(1)/source_ids.ne(1).sum(1)[:,None]#shape:batch_size*hidden_size
        outputs = outputs.reshape(-1,2,outputs.size(-1))#shape:batch_size/2 *2*hidden_size
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        return cos_sim #cos_sim, labels

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, 512)#[batch*2,512]

        if self.model_type in ['t5','codet5']:
            vec = self.get_t5_vec(source_ids)#[batch*2,768]
            logits = self.classifier(vec)#[batch,2]
            prob = nn.functional.softmax(logits)
        elif self.model_type in ['bart','plbart']:
            vec = self.get_bart_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.model_type in ['roberta', 'codebert', 'graphcodebert']:
            vec = self.get_roberta_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.model_type in ['unixcoder']:
            logits = self.get_unixcoder_vec(source_ids)
            prob = logits #=cos_sim

        if labels is not None:
            if self.model_type not in ['unixcoder']:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                loss = ((logits-labels.float())**2).mean()
                return loss, prob
        else:
            return prob


def convert_clone_examples_to_features(source,target,model_type):
    source = ' '.join(source.split())
    target = ' '.join(target.split())
    if model_type in ['t5', 'codet5']:
        source_str = "{}: {}".format('clone', source)
        target_str = "{}: {}".format('clone', target)
    elif model_type in ['unixcoder']:
        source_str = tokenizer.tokenize(source[:512-4])#format_special_chars(tokenizer.tokenize(source[:args.max_source_length-4]))
        source_str =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+source_str+[tokenizer.sep_token]
        target_str = tokenizer.tokenize(target[:512-4])#format_special_chars(tokenizer.tokenize(target[:args.max_target_length-4]))
        target_str =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+target_str+[tokenizer.sep_token]
        example_index = source_str + target_str
    else:

        source_str = source
        target_str = target
    if model_type in ['unixcoder']:
        code1 = tokenizer.convert_tokens_to_ids(source_str)
        padding_length = 512 - len(code1)
        code1 += [tokenizer.pad_token_id]*padding_length

        code2 = tokenizer.convert_tokens_to_ids(target_str)
        padding_length = 512 - len(code2)
        code2 += [tokenizer.pad_token_id]*padding_length
        source_ids = code1 + code2
    else:
        code1 = tokenizer.encode(
            source_str, max_length=512, padding='max_length', truncation=True)
        code2 = tokenizer.encode(
            target_str, max_length=512, padding='max_length', truncation=True)
        source_ids = code1 + code2
    return torch.tensor(
            [source_ids], dtype=torch.long)
# 加载预训练模型和分词器
model = PLBartForConditionalGeneration.from_pretrained('/root/autodl-tmp/HugCode/data/huggingface_models/plbart-base/')
config =PLBartConfig.from_pretrained(ckpt)
# model = PLBartForConditionalGeneration(config)

tokenizer = PLBartTokenizer.from_pretrained(ckpt)

model=CloneModel(model, config, tokenizer, model_type)
model = model.module if hasattr(model, 'module') else model
file = os.path.join(model_name, 'pytorch_model.bin')
model.load_state_dict(torch.load(file))
torch.manual_seed(1234)
# params1 = [name for name, param in list(model.named_parameters())]
params2 = [name for name, param in list(model.named_parameters())]

# diff=list(set(params1)^set(params2))
# print(diff)

# 将两段代码拼接并用 <sep> 分隔
inputs = convert_clone_examples_to_features(func1,func2,model_type)
# input_sequence = func1 + " <sep> " + func2

# # 使用分词器对输入序列进行编码
# inputs = tokenizer(input_sequence, return_tensors="pt")

# 将编码后的输入传递给模型
outputs = model(inputs)

# 获取分类概率
probs = outputs#torch.softmax(outputs.logits, dim=-1)
print(probs)
# 获取克隆概率
clone_prob = probs[0, 1].item()

print("Clone Probability:", clone_prob)
