'''
# -*- coding: utf-8 -*-
Author: nchen909 NuoChen
Date: 2023-05-07 16:59:19
FilePath: /HugNLP/applications/code/HugClone/clone_api.py
'''
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
import os
from processors.code.code_clone.data_processor import CodeCloneProcessor
from models import CODE_MODEL_CLASSES
from models import TOKENIZER_CLASSES
import torch
from torch import nn

class HugCloneAPI:
    def __init__(self, model_type, hugcode_model_name_or_path) -> None:
        if model_type not in CODE_MODEL_CLASSES["code_cls"].keys():
            raise KeyError(
                "You must choose one of the following model: {}".format(
                    ", ".join(
                        list(CODE_MODEL_CLASSES["code_cls"].
                             keys()))))
        self.model_type = model_type
        self.config =CODE_MODEL_CLASSES["code_cls"][self.model_type].from_pretrained(hugcode_model_name_or_path)
        self.tokenizer = TOKENIZER_CLASSES[self.model_type].from_pretrained(
            hugcode_model_name_or_path)
        self.model = CODE_MODEL_CLASSES["code_cls"][
            self.model_type](self.config).from_pretrained(hugcode_model_name_or_path)
        self.max_source_length = 512
        self.max_target_length = 512

    def request(self, func1: str, func2: str):
        examples = [{'label':'0','func1':func1,'func2':func2,'id':0}]
        processor = CodeCloneProcessor()
        preprocess_function = processor.build_preprocess_function()
        inputs= examples.map(
                preprocess_function,
                batched=True,
                desc="tokenize examples",
            )
        collator = processor.get_data_collator()
        batch_input=collator(inputs)
        # batch_input = {
        #     "input_ids": inputs["input_ids"],
        #     "attention_mask": inputs["attention_mask"],
        # }
        outputs = self.model(**batch_input)
        predictions, topk_result = processor.get_predict_result(outputs['logits'],examples, "test")
        clone_probability = predictions['prob']
        return clone_probability

if __name__ == "__main__":
    from applications.code.HugClone.clone_api import HugCloneAPI
    model_type = "plbart"
    hugclone_model_name_or_path = "/code/cn/HugAILab/HugNLP/outputs/code/clone/codebert-base/checkpoint-27300/"
    hugclone = HugCloneAPI(model_type, hugclone_model_name_or_path)

    ## JAVA code clone detection
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
    func2="""
    public static void copyFile(File in, File out) throws Exception {
        FileChannel sourceChannel = new FileInputStream(in).getChannel();
        FileChannel destinationChannel = new FileOutputStream(out).getChannel();
        sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
        sourceChannel.close();
        destinationChannel.close();
    }
    """
    clone_probability = hugclone.request(func1, func2)
    print("clone_probability:{}".format(clone_probability))
    print("\n\n")

    ## JAVA code clone detection
    func1="""
    public static void copyFile(File source, File dest) throws IOException {
        FileChannel in = null, out = null;
        try {
            in = new FileInputStream(source).getChannel();
            out = new FileOutputStream(dest).getChannel();
            in.transferTo(0, in.size(), out);
        } catch (FileNotFoundException fnfe) {
            Log.debug(fnfe);
        } finally {
            if (in != null) in.close();
            if (out != null) out.close();
        }
    }
    """

    func2="""
    public static void copyFile(File from, File to) throws IOException {
        if (from.isDirectory()) {
            if (!to.exists()) {
                to.mkdir();
            }
            File[] children = from.listFiles();
            for (int i = 0; i < children.length; i++) {
                if (children[i].getName().equals(".") || children[i].getName().equals("..")) {
                    continue;
                }
                if (children[i].isDirectory()) {
                    File f = new File(to, children[i].getName());
                    copyFile(children[i], f);
                } else {
                    copyFile(children[i], to);
                }
            }
        } else if (from.isFile() && (to.isDirectory() || to.isFile())) {
            if (to.isDirectory()) {
                to = new File(to, from.getName());
            }
            FileInputStream in = new FileInputStream(from);
            FileOutputStream out = new FileOutputStream(to);
            byte[] buf = new byte[32678];
            int read;
            while ((read = in.read(buf)) > -1) {
                out.write(buf, 0, read);
            }
            closeStream(in);
            closeStream(out);
        }
    }
    """
    clone_probability = hugclone.request(func1, func2)
    print("clone_probability:{}".format(clone_probability))

"""
clone_probability:2.0006775685033062e-06
clone_probability:0.9999953508377075
    """
